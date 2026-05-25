/*
 * Copyright (C) 2026 Au-Zone Technologies
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * edgefirst_json_extract.hpp — Read the edgefirst.json schema embedded
 * in a TFLite associated-files ZIP trailer.
 */

#pragma once

#include <optional>
#include <string>

/*
 * Extract the `edgefirst.json` entry embedded in a TFLite model's
 * associated-files ZIP trailer.
 *
 * EdgeFirst-converted TFLite models concatenate a ZIP archive to the
 * end of the FlatBuffer; the TFLite loader ignores trailing bytes and
 * the archive carries `edgefirst.json` (HAL decoder schema) plus
 * `labels.txt` (class names).
 *
 * Returns the decompressed JSON content on success.
 *
 * Returns std::nullopt when:
 *   - the model file can't be opened or read,
 *   - the file has no ZIP trailer (older model, plain TFLite),
 *   - the ZIP trailer contains no `edgefirst.json` entry, or
 *   - the entry exists but its content is corrupt.
 *
 * Callers should fall back to schema-less decoder construction
 * (decoder-version=yolov8 + tensor-shape inference) in these cases.
 * No exceptions; no logging — caller owns the user-facing message.
 */
std::optional<std::string>
extract_edgefirst_json_from_tflite(const std::string &model_path);
