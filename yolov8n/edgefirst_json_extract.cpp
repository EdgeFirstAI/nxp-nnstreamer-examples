/*
 * Copyright (C) 2026 Au-Zone Technologies
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * edgefirst_json_extract.cpp — minimal ZIP-trailer reader for TFLite
 * associated-files containers.
 *
 * The TFLite associated-files convention is to concatenate a standard
 * ZIP archive to the end of the FlatBuffer. Loaders ignore trailing
 * bytes. We mirror what `edgefirst_tflite::archive::ModelArchive` does
 * in Rust (and what `~/profiler/src/inference/tflite.rs` uses): scan
 * backwards for the End-of-Central-Directory record, walk the central
 * directory to find the `edgefirst.json` entry, then inflate it via
 * zlib's raw-deflate mode.
 *
 * Only the two compression methods that EdgeFirst Studio's exporter
 * actually produces (STORE / DEFLATE) are supported. Any anomaly
 * along the way returns std::nullopt so the caller falls back to the
 * legacy tensor-shape decoder configuration.
 */

#include "edgefirst_json_extract.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>

#include <zlib.h>

namespace {

/* ZIP record signatures (little-endian on disk). */
constexpr uint32_t SIG_EOCD = 0x06054b50;  /* End of Central Directory. */
constexpr uint32_t SIG_CDFH = 0x02014b50;  /* Central Directory File Header. */
constexpr uint32_t SIG_LFH  = 0x04034b50;  /* Local File Header. */

/* Compression methods we accept. */
constexpr uint16_t METHOD_STORE   = 0;
constexpr uint16_t METHOD_DEFLATE = 8;

/* Little-endian field readers. ZIP fields are always LE regardless of host. */
inline uint16_t rd16(const uint8_t *p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
inline uint32_t rd32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

/*
 * Read the whole file into memory. Returns an empty vector on any I/O
 * failure. TFLite models are 1–50 MB, so single-shot read is fine.
 */
std::vector<uint8_t> read_whole_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const std::streamsize sz = f.tellg();
  if (sz <= 0) return {};
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  if (!f.read(reinterpret_cast<char *>(buf.data()), sz)) return {};
  return buf;
}

/*
 * Find the End-of-Central-Directory record by scanning backwards from
 * the last possible EOCD start. The minimum EOCD size is 22 bytes and
 * the optional trailing comment can be up to 65 535 bytes, so we scan
 * at most ~64 KiB from the end of the file.
 *
 * Returns the byte offset of the EOCD signature, or -1 if absent.
 */
ssize_t find_eocd(const std::vector<uint8_t> &data) {
  constexpr size_t MIN_EOCD = 22;
  constexpr size_t MAX_CMT  = 65535;
  if (data.size() < MIN_EOCD) return -1;
  const size_t last_start = data.size() - MIN_EOCD;
  const size_t scan_max = std::min(last_start, MAX_CMT);
  for (size_t back = 0; back <= scan_max; ++back) {
    const size_t i = last_start - back;
    if (rd32(&data[i]) == SIG_EOCD) {
      return static_cast<ssize_t>(i);
    }
  }
  return -1;
}

/*
 * Inflate `src` (raw DEFLATE, no zlib/gzip header) into `out`.
 * Pre-sized `out` to the expected uncompressed size before calling.
 */
bool inflate_deflate(const uint8_t *src, size_t src_len,
                     size_t uncompressed_size, std::string &out) {
  out.resize(uncompressed_size);
  z_stream zs{};
  zs.next_in   = const_cast<Bytef *>(src);
  zs.avail_in  = static_cast<uInt>(src_len);
  zs.next_out  = reinterpret_cast<Bytef *>(out.data());
  zs.avail_out = static_cast<uInt>(uncompressed_size);
  /* Negative window bits => raw deflate (no zlib/gzip wrapper). */
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false;
  const int r = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  return r == Z_STREAM_END && zs.total_out == uncompressed_size;
}

} /* namespace */

std::optional<std::string>
extract_edgefirst_json_from_tflite(const std::string &model_path) {
  const std::vector<uint8_t> data = read_whole_file(model_path);
  if (data.empty()) return std::nullopt;

  const ssize_t eocd_off = find_eocd(data);
  if (eocd_off < 0) return std::nullopt;

  /*
   * EOCD record fields (22 bytes minimum):
   *    0  uint32  signature (0x06054b50)
   *    4  uint16  disk number
   *    6  uint16  disk where CD starts
   *    8  uint16  CD records on this disk
   *   10  uint16  total CD records
   *   12  uint32  size of CD in bytes
   *   16  uint32  offset of CD start (from beginning of archive)
   *   20  uint16  comment length
   */
  const uint8_t *eocd = &data[eocd_off];
  const uint16_t cd_count  = rd16(eocd + 10);
  const uint32_t cd_size   = rd32(eocd + 12);
  const uint32_t cd_offset = rd32(eocd + 16);

  /*
   * The archive may sit at a non-zero offset within the file (it does
   * when appended to a FlatBuffer). The CD offset stored in the EOCD
   * is from the start of the archive, but ZIP libraries reinterpret it
   * as a file offset; the spec allows the discrepancy and most readers
   * (including ours) accept the value as-is when it points to a valid
   * CDFH signature.
   */
  if (static_cast<size_t>(cd_offset) + cd_size > static_cast<size_t>(eocd_off)) {
    return std::nullopt;
  }

  /*
   * Walk Central Directory entries. Each is at least 46 bytes:
   *    0  uint32  signature (0x02014b50)
   *    4  uint16  version made by
   *    6  uint16  version needed
   *    8  uint16  general purpose bit flag
   *   10  uint16  compression method
   *   12  uint16  last mod time
   *   14  uint16  last mod date
   *   16  uint32  CRC-32
   *   20  uint32  compressed size
   *   24  uint32  uncompressed size
   *   28  uint16  file name length
   *   30  uint16  extra field length
   *   32  uint16  file comment length
   *   34  uint16  disk number start
   *   36  uint16  internal file attrs
   *   38  uint32  external file attrs
   *   42  uint32  local file header offset
   *   46  ...     file name + extra + comment
   */
  size_t off = cd_offset;
  for (uint16_t i = 0; i < cd_count; ++i) {
    if (off + 46 > static_cast<size_t>(eocd_off)) return std::nullopt;
    if (rd32(&data[off]) != SIG_CDFH) return std::nullopt;

    const uint16_t method      = rd16(&data[off + 10]);
    const uint32_t comp_size   = rd32(&data[off + 20]);
    const uint32_t uncomp_size = rd32(&data[off + 24]);
    const uint16_t name_len    = rd16(&data[off + 28]);
    const uint16_t extra_len   = rd16(&data[off + 30]);
    const uint16_t cmt_len     = rd16(&data[off + 32]);
    const uint32_t lfh_off     = rd32(&data[off + 42]);

    if (off + 46 + name_len > static_cast<size_t>(eocd_off)) {
      return std::nullopt;
    }
    const std::string name(reinterpret_cast<const char *>(&data[off + 46]),
                           name_len);

    const size_t next_off = off + 46 + name_len + extra_len + cmt_len;

    if (name == "edgefirst.json") {
      /*
       * Local File Header fields we need (30 bytes minimum):
       *    0  uint32  signature (0x04034b50)
       *   26  uint16  file name length
       *   28  uint16  extra field length
       *   30  ...     name + extra + compressed data
       */
      if (lfh_off + 30 > data.size()) return std::nullopt;
      if (rd32(&data[lfh_off]) != SIG_LFH) return std::nullopt;

      const uint16_t lfh_name_len  = rd16(&data[lfh_off + 26]);
      const uint16_t lfh_extra_len = rd16(&data[lfh_off + 28]);
      const size_t data_off = lfh_off + 30 + lfh_name_len + lfh_extra_len;

      if (data_off + comp_size > data.size()) return std::nullopt;

      std::string result;
      if (method == METHOD_STORE) {
        result.assign(reinterpret_cast<const char *>(&data[data_off]),
                      comp_size);
        return result;
      }
      if (method == METHOD_DEFLATE) {
        if (!inflate_deflate(&data[data_off], comp_size, uncomp_size, result)) {
          return std::nullopt;
        }
        return result;
      }
      /* Unknown compression method — give up. */
      return std::nullopt;
    }

    off = next_off;
  }

  return std::nullopt;
}
