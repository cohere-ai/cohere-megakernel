/*
 * SM-level Profiler Library - Host implementation
 *
 * Buffer layout (all uint64-aligned):
 *   [header:   2 x uint64]
 *   [counters: num_blocks * num_groups x uint32, padded to uint64]
 *   [events:   num_blocks * num_groups * max_events x 24-byte SmProfilerDeviceEvent]
 *
 * Adapted from https://github.com/leepoly/sm-profiler; see README.md in this
 * directory for the changes.
 */

#include "sm_profiler.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static constexpr uint32_t HEADER_SIZE_UINT64 = 2;
static constexpr size_t   DEVICE_EVENT_SIZE  = 24;

static_assert(sizeof(SmProfilerDeviceEvent) == DEVICE_EVENT_SIZE,
              "SmProfilerDeviceEvent must be 24 bytes");

struct sm_profiler_buffer {
    uint32_t num_blocks{};
    uint32_t num_groups{};
    uint32_t max_events_per_group{};
    bool enabled{};
    bool owns_device_memory{};
    uint64_t* device_ptr{};

    std::vector<uint64_t>    host_copy;
    std::vector<std::string> event_names;

    size_t buffer_bytes() const { return host_copy.size() * sizeof(uint64_t); }

    ~sm_profiler_buffer() {
        if (device_ptr && owns_device_memory) cudaFree(device_ptr);
    }

    sm_profiler_buffer() = default;
    sm_profiler_buffer(const sm_profiler_buffer&) = delete;
    sm_profiler_buffer& operator=(const sm_profiler_buffer&) = delete;
};

/* ---- Layout helpers ---- */

static uint32_t get_counters_size_uint64(uint32_t num_blocks, uint32_t num_groups) {
    auto counters_bytes = static_cast<size_t>(num_blocks) * num_groups * sizeof(uint32_t);
    return static_cast<uint32_t>((counters_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
}

static uint32_t get_event_data_offset(uint32_t num_blocks, uint32_t num_groups) {
    return HEADER_SIZE_UINT64 + get_counters_size_uint64(num_blocks, num_groups);
}

/* Returns buffer size in uint64 units */
static size_t calc_buffer_size_u64(
    uint32_t num_blocks, uint32_t num_groups,
    uint32_t max_events_per_group, bool enabled
) {
    if (!enabled) return HEADER_SIZE_UINT64;
    auto event_offset_u64 = get_event_data_offset(num_blocks, num_groups);
    auto event_data_bytes = static_cast<size_t>(num_blocks) * num_groups
                            * max_events_per_group * DEVICE_EVENT_SIZE;
    auto event_data_u64 = (event_data_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    return event_offset_u64 + event_data_u64;
}

/* ---- Public API ---- */

size_t sm_profiler_calc_buffer_size(
    uint32_t num_blocks, uint32_t num_groups,
    uint32_t max_events_per_group, int enabled
) {
    if (num_blocks == 0 || num_groups == 0 || max_events_per_group == 0) return 0;
    return calc_buffer_size_u64(num_blocks, num_groups, max_events_per_group, enabled)
           * sizeof(uint64_t);
}

sm_profiler_buffer_t sm_profiler_create_buffer_external(
    uint64_t* external_device_ptr,
    size_t buffer_size,
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
) {
    if (!external_device_ptr || num_blocks == 0 || num_groups == 0 || max_events_per_group == 0) {
        std::cerr << "sm_profiler: Invalid parameters for external buffer\n";
        return nullptr;
    }

    auto required_u64 = calc_buffer_size_u64(num_blocks, num_groups, max_events_per_group, enabled);
    auto required_bytes = required_u64 * sizeof(uint64_t);
    if (buffer_size < required_bytes) {
        std::cerr << "sm_profiler: External buffer too small ("
                  << buffer_size << " < " << required_bytes << ")\n";
        return nullptr;
    }

    auto buf = std::make_unique<sm_profiler_buffer>();
    buf->num_blocks = num_blocks;
    buf->num_groups = num_groups;
    buf->max_events_per_group = max_events_per_group;
    buf->enabled = enabled;
    buf->owns_device_memory = false;
    buf->device_ptr = external_device_ptr;
    buf->host_copy.resize(required_u64, 0);

    return buf.release();
}

sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
) {
    if (num_blocks == 0 || num_groups == 0 || max_events_per_group == 0) {
        std::cerr << "sm_profiler: Invalid parameters\n";
        return nullptr;
    }

    auto buf = std::make_unique<sm_profiler_buffer>();
    buf->num_blocks = num_blocks;
    buf->num_groups = num_groups;
    buf->max_events_per_group = max_events_per_group;
    buf->enabled = enabled;
    buf->owns_device_memory = true;

    auto size_u64  = calc_buffer_size_u64(num_blocks, num_groups, max_events_per_group, enabled);
    auto size_bytes = size_u64 * sizeof(uint64_t);

    cudaError_t err = cudaMalloc(&buf->device_ptr, size_bytes);
    if (err != cudaSuccess) {
        std::cerr << "sm_profiler: cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        buf->device_ptr = nullptr;
        return nullptr;
    }

    err = cudaMemset(buf->device_ptr, 0, size_bytes);
    if (err != cudaSuccess) {
        std::cerr << "sm_profiler: cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    buf->host_copy.resize(size_u64, 0);

    return buf.release();
}

void sm_profiler_destroy_buffer(sm_profiler_buffer_t buffer) {
    delete buffer;
}

uint64_t* sm_profiler_get_device_ptr(sm_profiler_buffer_t buffer) {
    return buffer ? buffer->device_ptr : nullptr;
}

void sm_profiler_init_buffer(sm_profiler_buffer_t buffer) {
    if (!buffer) return;

    buffer->host_copy[0] = (static_cast<uint64_t>(buffer->num_blocks) << 32)
                         | static_cast<uint64_t>(buffer->num_groups);
    buffer->host_copy[1] = (static_cast<uint64_t>(buffer->max_events_per_group) << 32)
                         | static_cast<uint64_t>(buffer->enabled ? 1 : 0);

    if (buffer->enabled) {
        std::fill(buffer->host_copy.begin() + HEADER_SIZE_UINT64,
                  buffer->host_copy.end(), uint64_t{0});
    }

    cudaError_t err = cudaMemcpy(buffer->device_ptr, buffer->host_copy.data(),
                                 buffer->buffer_bytes(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cerr << "sm_profiler: cudaMemcpy failed in init_buffer: "
                  << cudaGetErrorString(err) << "\n";
}

int sm_profiler_register_event(
    sm_profiler_buffer_t buffer,
    uint32_t event_no,
    const char* name
) {
    if (!buffer || !name) return -1;
    if (event_no >= buffer->event_names.size())
        buffer->event_names.resize(event_no + 1);
    buffer->event_names[event_no] = name;
    return 0;
}

void sm_profiler_get_info(
    sm_profiler_buffer_t buffer,
    uint32_t* out_num_blocks,
    uint32_t* out_num_groups,
    uint32_t* out_max_events
) {
    if (!buffer) return;
    if (out_num_blocks) *out_num_blocks = buffer->num_blocks;
    if (out_num_groups) *out_num_groups = buffer->num_groups;
    if (out_max_events) *out_max_events = buffer->max_events_per_group;
}

/* ============================================================================
 * Export to Chrome Trace JSON format
 * ============================================================================ */

struct BufferHeader {
    uint32_t num_blocks;
    uint32_t num_groups;
    uint32_t max_events_per_group;
};

static BufferHeader parse_header(const uint64_t* data) {
    return {
        static_cast<uint32_t>(data[0] >> 32),
        static_cast<uint32_t>(data[0] & 0xFFFFFFFF),
        static_cast<uint32_t>(data[1] >> 32)
    };
}

static const SmProfilerDeviceEvent* get_device_event(
    const void* data_ptr, uint32_t num_groups, uint32_t max_events_per_group,
    uint32_t block_idx, uint32_t group_idx, uint32_t event_idx
) {
    auto group_offset = (static_cast<size_t>(block_idx) * num_groups + group_idx)
                        * max_events_per_group;
    return reinterpret_cast<const SmProfilerDeviceEvent*>(
        static_cast<const char*>(data_ptr) + (group_offset + event_idx) * DEVICE_EVENT_SIZE);
}

struct ExportEvent {
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
    uint32_t event_no;
    uint32_t sm_id;
    uint32_t block_id;
    uint32_t group_idx;
    uint32_t event_id;
    uint32_t type;
};

static int sm_profiler_export_to_file_impl(
    sm_profiler_buffer_t buffer,
    const char* filename,
    bool compact
) {
    if (!buffer || !filename) {
        std::cerr << "sm_profiler_export_to_file: Invalid arguments\n";
        return -1;
    }

    cudaError_t err = cudaMemcpy(buffer->host_copy.data(), buffer->device_ptr,
                                 buffer->buffer_bytes(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "sm_profiler: cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    auto hdr = parse_header(buffer->host_copy.data());

    if (hdr.num_blocks == 0 || hdr.num_groups == 0 || hdr.max_events_per_group == 0) {
        std::cerr << "sm_profiler: Invalid header in buffer\n";
        return -1;
    }
    if (!buffer->enabled) {
        std::cerr << "sm_profiler: Profiling is disabled, no events to export\n";
        return -1;
    }

    auto* counters = reinterpret_cast<const uint32_t*>(
        buffer->host_copy.data() + HEADER_SIZE_UINT64);
    auto ev_off = get_event_data_offset(hdr.num_blocks, hdr.num_groups);
    auto* event_data_ptr = reinterpret_cast<const char*>(buffer->host_copy.data())
                           + ev_off * sizeof(uint64_t);

    size_t num_counters = static_cast<size_t>(hdr.num_blocks) * hdr.num_groups;
    size_t total_events = 0;
    for (size_t i = 0; i < num_counters; i++)
        total_events += counters[i];

    if (total_events == 0) {
        std::cerr << "sm_profiler: No events recorded\n";
        return -1;
    }

    std::vector<ExportEvent> events;
    events.reserve(total_events);

    for (uint32_t b = 0; b < hdr.num_blocks; b++) {
        for (uint32_t g = 0; g < hdr.num_groups; g++) {
            uint32_t cnt = counters[b * hdr.num_groups + g];
            for (uint32_t e = 0; e < cnt && e < hdr.max_events_per_group; e++) {
                auto* dev = get_device_event(
                    event_data_ptr, hdr.num_groups, hdr.max_events_per_group, b, g, e);
                events.push_back({
                    dev->st_timestamp_ns,
                    dev->en_timestamp_ns,
                    dev->event_no & SM_PROFILER_EVENT_NO_MASK,
                    dev->sm_id,
                    b, g, e,
                    (dev->event_no & SM_PROFILER_EVENT_FLAG_INSTANT) ? 1u : 0u
                });
            }
        }
    }

    std::sort(events.begin(), events.end(), [](const ExportEvent& a, const ExportEvent& b) {
        return a.st_timestamp_ns < b.st_timestamp_ns;
    });

    uint64_t min_timestamp = events.empty() ? 0 : events.front().st_timestamp_ns;

    std::unique_ptr<FILE, decltype(&fclose)> fp(fopen(filename, "w"), &fclose);
    if (!fp) {
        std::cerr << "sm_profiler: Failed to open " << filename << " for writing\n";
        return -1;
    }

    fprintf(fp.get(), "{\n  \"traceEvents\": [\n");

    size_t output_count = 0;
    bool first_entry = true;

    for (const auto& ev : events) {
        std::string base_name = "unknown";
        if (ev.event_no < buffer->event_names.size() &&
            !buffer->event_names[ev.event_no].empty())
            base_name = buffer->event_names[ev.event_no];

        if (compact) {
            // Compact mode intentionally contains ranges only. An unfinished
            // range has no meaningful duration and is omitted rather than
            // creating a huge underflowed slice in Perfetto.
            if (ev.type != 0 || ev.en_timestamp_ns < ev.st_timestamp_ns) continue;

            auto tid = "block_" + std::to_string(ev.block_id);
            double ts_us =
                static_cast<double>(ev.st_timestamp_ns - min_timestamp) / 1000.0;
            double dur_us =
                static_cast<double>(ev.en_timestamp_ns - ev.st_timestamp_ns) / 1000.0;

            if (!first_entry) fprintf(fp.get(), ",\n");
            first_entry = false;
            fprintf(fp.get(),
                "    {\"name\": \"%s\", \"ph\": \"X\", \"ts\": %.3f, "
                "\"dur\": %.3f, \"pid\": 0, \"tid\": \"%s\", "
                "\"cat\": \"gpu\", \"args\": {\"sm_id\": %u}}",
                base_name.c_str(), ts_us, dur_us, tid.c_str(), ev.sm_id);
            output_count++;
            continue;
        }

        auto name_id = base_name + "_" + std::to_string(ev.event_id);
        auto tid = "block_" + std::to_string(ev.block_id)
                 + "_group_" + std::to_string(ev.group_idx);

        double ts_us = static_cast<double>(ev.st_timestamp_ns - min_timestamp) / 1000.0;

        if (ev.type == 0) {
            if (!first_entry) fprintf(fp.get(), ",\n");
            first_entry = false;

            fprintf(fp.get(),
                "    {\"name\": \"%s\", \"ph\": \"B\", \"ts\": %.3f, "
                "\"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\"}",
                name_id.c_str(), ts_us, ev.sm_id, tid.c_str());
            output_count++;

            double end_us = static_cast<double>(ev.en_timestamp_ns - min_timestamp) / 1000.0;
            fprintf(fp.get(), ",\n"
                "    {\"name\": \"%s\", \"ph\": \"E\", \"ts\": %.3f, "
                "\"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\"}",
                name_id.c_str(), end_us, ev.sm_id, tid.c_str());
            output_count++;
        } else {
            if (!first_entry) fprintf(fp.get(), ",\n");
            first_entry = false;

            fprintf(fp.get(),
                "    {\"name\": \"%s\", \"ph\": \"i\", \"ts\": %.3f, "
                "\"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\", \"s\": \"t\"}",
                name_id.c_str(), ts_us, ev.sm_id, tid.c_str());
            output_count++;
        }
    }

    fprintf(fp.get(), "\n  ],\n");
    fprintf(fp.get(), "  \"displayTimeUnit\": \"ns\",\n");
    fprintf(fp.get(), "  \"metadata\": {\n");
    fprintf(fp.get(), "    \"num_blocks\": %u,\n", hdr.num_blocks);
    fprintf(fp.get(), "    \"num_groups\": %u,\n", hdr.num_groups);
    fprintf(fp.get(), "    \"max_events_per_group\": %u,\n", hdr.max_events_per_group);
    fprintf(fp.get(), "    \"total_events\": %zu\n",
            compact ? output_count : events.size());
    fprintf(fp.get(), "  }\n}\n");

    // std::cout << "sm_profiler: Exported " << events.size() << " events ("
    //           << output_count << " trace entries) to " << filename << "\n";
    return 0;
}

int sm_profiler_export_to_file(
    sm_profiler_buffer_t buffer,
    const char* filename
) {
    return sm_profiler_export_to_file_impl(buffer, filename, false);
}

int sm_profiler_export_to_file_compact(
    sm_profiler_buffer_t buffer,
    const char* filename
) {
    return sm_profiler_export_to_file_impl(buffer, filename, true);
}
