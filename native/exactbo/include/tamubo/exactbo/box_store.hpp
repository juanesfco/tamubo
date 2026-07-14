#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tamubo::exactbo {

class BoxStore {
public:
    virtual ~BoxStore() = default;
    virtual std::uint64_t rows() const noexcept = 0;
    virtual std::uint64_t dims() const noexcept = 0;
    virtual bool file_backed() const noexcept = 0;
    virtual const std::string& path() const noexcept = 0;
    virtual void read_rows(std::uint64_t offset, std::uint64_t count,
                           double* lower, double* upper) const = 0;
};

class HostBoxStore final : public BoxStore {
public:
    HostBoxStore(std::uint64_t rows, std::uint64_t dims,
                 std::vector<double> lower, std::vector<double> upper);
    std::uint64_t rows() const noexcept override { return rows_; }
    std::uint64_t dims() const noexcept override { return dims_; }
    bool file_backed() const noexcept override { return false; }
    const std::string& path() const noexcept override { return empty_path_; }
    void read_rows(std::uint64_t offset, std::uint64_t count,
                   double* lower, double* upper) const override;
    const std::vector<double>& lower() const noexcept { return lower_; }
    const std::vector<double>& upper() const noexcept { return upper_; }

private:
    std::uint64_t rows_;
    std::uint64_t dims_;
    std::vector<double> lower_;
    std::vector<double> upper_;
    std::string empty_path_;
};

class FileBoxStore final : public BoxStore {
public:
    explicit FileBoxStore(std::string path);
    ~FileBoxStore() override;
    FileBoxStore(const FileBoxStore&) = delete;
    FileBoxStore& operator=(const FileBoxStore&) = delete;
    std::uint64_t rows() const noexcept override { return rows_; }
    std::uint64_t dims() const noexcept override { return dims_; }
    bool file_backed() const noexcept override { return true; }
    const std::string& path() const noexcept override { return path_; }
    void read_rows(std::uint64_t offset, std::uint64_t count,
                   double* lower, double* upper) const override;

private:
    std::string path_;
    int fd_ = -1;
    std::uint64_t rows_ = 0;
    std::uint64_t dims_ = 0;
};

class FileBoxWriter {
public:
    explicit FileBoxWriter(std::string path);
    ~FileBoxWriter();
    FileBoxWriter(const FileBoxWriter&) = delete;
    FileBoxWriter& operator=(const FileBoxWriter&) = delete;
    void write_rows(std::uint64_t offset, std::uint64_t count,
                    const double* lower, const double* upper);
    // Make every completed write durable, then ask the kernel to evict the
    // now-clean file pages so a spill file does not remain resident in RAM.
    void sync_and_drop_cache();

private:
    std::string path_;
    int fd_ = -1;
    std::uint64_t rows_ = 0;
    std::uint64_t dims_ = 0;
};

void initialize_box_file(const std::string& partial_path,
                         std::uint64_t rows, std::uint64_t dims);
void finalize_box_file(const std::string& partial_path,
                       const std::string& final_path);
std::uint64_t box_data_bytes(std::uint64_t rows, std::uint64_t dims);
std::uint64_t available_host_memory_bytes();
std::uint64_t available_filesystem_bytes(const std::string& directory);

}  // namespace tamubo::exactbo
