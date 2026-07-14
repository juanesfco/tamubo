#include "tamubo/exactbo/box_store.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <fcntl.h>
#ifdef __linux__
#include <linux/fs.h>
#include <sys/syscall.h>
#endif
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace tamubo::exactbo {
namespace {

constexpr std::array<char, 8> kMagic{'T', 'B', 'O', 'X', 'V', '1', '!', '!'};
constexpr std::uint64_t kHeaderBytes = 4096;
constexpr std::uint64_t kRowsOffset = 8;
constexpr std::uint64_t kDimsOffset = 16;

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(std::string(label) + " overflows uint64");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::runtime_error(std::string(label) + " overflows uint64");
    }
    return a + b;
}

off_t checked_offset(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds off_t");
    }
    return static_cast<off_t>(value);
}

void throw_errno(const std::string& action, const std::string& path) {
    throw std::runtime_error(action + " " + path + ": " + std::strerror(errno));
}

void pread_all(int fd, void* dst, std::uint64_t bytes, std::uint64_t offset,
               const std::string& path) {
    auto* out = static_cast<unsigned char*>(dst);
    while (bytes != 0) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(bytes, static_cast<std::uint64_t>(SSIZE_MAX)));
        ssize_t got = ::pread(fd, out, chunk, checked_offset(offset, "file read offset"));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got < 0) {
            throw_errno("failed to read", path);
        }
        if (got == 0) {
            throw std::runtime_error("unexpected end of box file: " + path);
        }
        out += got;
        bytes -= static_cast<std::uint64_t>(got);
        offset += static_cast<std::uint64_t>(got);
    }
}

void pwrite_all(int fd, const void* src, std::uint64_t bytes, std::uint64_t offset,
                const std::string& path) {
    const auto* in = static_cast<const unsigned char*>(src);
    while (bytes != 0) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(bytes, static_cast<std::uint64_t>(SSIZE_MAX)));
        ssize_t put = ::pwrite(fd, in, chunk, checked_offset(offset, "file write offset"));
        if (put < 0 && errno == EINTR) {
            continue;
        }
        if (put < 0) {
            throw_errno("failed to write", path);
        }
        if (put == 0) {
            throw std::runtime_error("short write to box file: " + path);
        }
        in += put;
        bytes -= static_cast<std::uint64_t>(put);
        offset += static_cast<std::uint64_t>(put);
    }
}

void advise_dontneed(int fd, std::uint64_t offset, std::uint64_t bytes,
                     const std::string& path) {
#ifdef POSIX_FADV_DONTNEED
    if (bytes == 0) {
        return;
    }
    const int status = ::posix_fadvise(
        fd, checked_offset(offset, "cache-advice offset"),
        checked_offset(bytes, "cache-advice length"), POSIX_FADV_DONTNEED);
    if (status != 0 && status != EINVAL && status != ENOSYS
#ifdef EOPNOTSUPP
        && status != EOPNOTSUPP
#endif
    ) {
        errno = status;
        throw_errno("failed to drop cached pages for", path);
    }
#else
    (void)fd;
    (void)offset;
    (void)bytes;
    (void)path;
#endif
}

void close_checked(int fd, const std::string& path) {
    if (::close(fd) != 0) {
        throw_errno("failed to close", path);
    }
}

void rename_noreplace(const std::string& source, const std::string& destination) {
#if defined(__linux__) && defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                  destination.c_str(), RENAME_NOREPLACE) == 0) {
        return;
    }
    const int rename_error = errno;
    if (rename_error != ENOSYS && rename_error != EINVAL
#ifdef EOPNOTSUPP
        && rename_error != EOPNOTSUPP
#endif
    ) {
        errno = rename_error;
        throw_errno("failed to finalize " + source + " as", destination);
    }
#endif

    // link()+unlink() is an atomic no-replace publication on the same
    // filesystem and is the fallback when renameat2 is unavailable or the
    // shared filesystem does not implement it.
    if (::link(source.c_str(), destination.c_str()) != 0) {
        throw_errno("failed to finalize " + source + " as", destination);
    }
    if (::unlink(source.c_str()) != 0) {
        const int unlink_error = errno;
        ::unlink(destination.c_str());
        errno = unlink_error;
        throw_errno("failed to remove finalized partial file", source);
    }
}

void sync_parent_directory(const std::string& path) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) {
        parent = ".";
    }
    const std::string parent_string = parent.string();
    const int fd = ::open(parent_string.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw_errno("failed to open parent directory", parent_string);
    }
    if (::fsync(fd) != 0) {
        const int sync_error = errno;
        ::close(fd);
        errno = sync_error;
        throw_errno("failed to sync parent directory", parent_string);
    }
    close_checked(fd, parent_string);
}

std::uint64_t lower_offset(std::uint64_t row, std::uint64_t dims) {
    const std::uint64_t bytes = checked_mul(
        checked_mul(row, dims, "row elements"), sizeof(double), "row bytes");
    return checked_add(kHeaderBytes, bytes, "lower offset");
}

std::uint64_t upper_offset(std::uint64_t rows, std::uint64_t row,
                           std::uint64_t dims) {
    const std::uint64_t plane = checked_mul(
        checked_mul(rows, dims, "plane elements"), sizeof(double), "plane bytes");
    return checked_add(
        checked_add(kHeaderBytes, plane, "upper plane offset"),
        checked_mul(checked_mul(row, dims, "row elements"), sizeof(double), "row bytes"),
        "upper offset");
}

void read_header(int fd, const std::string& path,
                 std::uint64_t& rows, std::uint64_t& dims) {
    std::array<char, 8> magic{};
    pread_all(fd, magic.data(), magic.size(), 0, path);
    if (magic != kMagic) {
        throw std::runtime_error("invalid ExactBO box-store magic: " + path);
    }
    pread_all(fd, &rows, sizeof(rows), kRowsOffset, path);
    pread_all(fd, &dims, sizeof(dims), kDimsOffset, path);
    if (dims == 0) {
        throw std::runtime_error("box-store dimension must be positive: " + path);
    }
    const std::uint64_t expected =
        checked_add(kHeaderBytes, box_data_bytes(rows, dims), "box file size");
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        throw_errno("failed to stat", path);
    }
    if (st.st_size < 0 || static_cast<std::uint64_t>(st.st_size) != expected) {
        throw std::runtime_error("box-store file has the wrong size: " + path);
    }
}

std::uint64_t read_uint_file(const std::filesystem::path& path, bool& finite) {
    std::ifstream in(path);
    std::string value;
    if (!(in >> value) || value == "max") {
        finite = false;
        return std::numeric_limits<std::uint64_t>::max();
    }
    try {
        finite = true;
        return std::stoull(value);
    } catch (...) {
        finite = false;
        return std::numeric_limits<std::uint64_t>::max();
    }
}

std::filesystem::path current_cgroup_v2_directory() {
    std::ifstream in("/proc/self/cgroup");
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t separator = line.find("::");
        if (separator == std::string::npos) {
            continue;
        }
        std::filesystem::path relative = line.substr(separator + 2);
        if (relative.is_absolute()) {
            relative = relative.relative_path();
        }
        return std::filesystem::path("/sys/fs/cgroup") / relative;
    }
    return std::filesystem::path("/sys/fs/cgroup");
}

std::filesystem::path current_cgroup_v1_memory_directory() {
    std::ifstream in("/proc/self/cgroup");
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t first = line.find(':');
        const std::size_t second =
            first == std::string::npos ? first : line.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            continue;
        }
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string padded = "," + controllers + ",";
        if (padded.find(",memory,") == std::string::npos) {
            continue;
        }
        std::filesystem::path relative = line.substr(second + 1);
        if (relative.is_absolute()) {
            relative = relative.relative_path();
        }
        return std::filesystem::path("/sys/fs/cgroup/memory") / relative;
    }
    return std::filesystem::path("/sys/fs/cgroup/memory");
}

}  // namespace

std::uint64_t box_data_bytes(std::uint64_t rows, std::uint64_t dims) {
    return checked_mul(
        checked_mul(checked_mul(rows, dims, "box elements"), 2, "lower/upper elements"),
        sizeof(double), "box bytes");
}

HostBoxStore::HostBoxStore(std::uint64_t rows, std::uint64_t dims,
                           std::vector<double> lower,
                           std::vector<double> upper)
    : rows_(rows), dims_(dims), lower_(std::move(lower)), upper_(std::move(upper)) {
    const std::uint64_t expected = checked_mul(rows_, dims_, "host box elements");
    if (dims_ == 0 || lower_.size() != expected || upper_.size() != expected) {
        throw std::runtime_error("invalid HostBoxStore shape");
    }
}

void HostBoxStore::read_rows(std::uint64_t offset, std::uint64_t count,
                             double* lower, double* upper) const {
    if (offset > rows_ || count > rows_ - offset) {
        throw std::runtime_error("HostBoxStore read is out of range");
    }
    const std::uint64_t begin = checked_mul(offset, dims_, "host read offset");
    const std::uint64_t elements = checked_mul(count, dims_, "host read elements");
    std::copy_n(lower_.data() + begin, elements, lower);
    std::copy_n(upper_.data() + begin, elements, upper);
}

FileBoxStore::FileBoxStore(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        throw_errno("failed to open", path_);
    }
    try {
        read_header(fd_, path_, rows_, dims_);
    } catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

FileBoxStore::~FileBoxStore() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void FileBoxStore::read_rows(std::uint64_t offset, std::uint64_t count,
                             double* lower, double* upper) const {
    if (offset > rows_ || count > rows_ - offset) {
        throw std::runtime_error("FileBoxStore read is out of range");
    }
    const std::uint64_t bytes = checked_mul(
        checked_mul(count, dims_, "file read elements"), sizeof(double),
        "file read bytes");
    pread_all(fd_, lower, bytes, lower_offset(offset, dims_), path_);
    pread_all(fd_, upper, bytes, upper_offset(rows_, offset, dims_), path_);
#ifdef POSIX_FADV_DONTNEED
    ::posix_fadvise(fd_, checked_offset(lower_offset(offset, dims_), "lower advice offset"),
                    checked_offset(bytes, "advice length"), POSIX_FADV_DONTNEED);
    ::posix_fadvise(fd_, checked_offset(upper_offset(rows_, offset, dims_), "upper advice offset"),
                    checked_offset(bytes, "advice length"), POSIX_FADV_DONTNEED);
#endif
}

FileBoxWriter::FileBoxWriter(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        throw_errno("failed to open", path_);
    }
    try {
        read_header(fd_, path_, rows_, dims_);
    } catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

FileBoxWriter::~FileBoxWriter() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void FileBoxWriter::write_rows(std::uint64_t offset, std::uint64_t count,
                               const double* lower, const double* upper) {
    if (offset > rows_ || count > rows_ - offset) {
        throw std::runtime_error("FileBoxWriter write is out of range");
    }
    const std::uint64_t bytes = checked_mul(
        checked_mul(count, dims_, "file write elements"), sizeof(double),
        "file write bytes");
    const std::uint64_t lower_file_offset = lower_offset(offset, dims_);
    const std::uint64_t upper_file_offset = upper_offset(rows_, offset, dims_);
    pwrite_all(fd_, lower, bytes, lower_file_offset, path_);
    pwrite_all(fd_, upper, bytes, upper_file_offset, path_);
    // On Linux this can start writeback immediately. Dirty pages are not
    // guaranteed to be evicted until sync_and_drop_cache() makes them clean.
    advise_dontneed(fd_, lower_file_offset, bytes, path_);
    advise_dontneed(fd_, upper_file_offset, bytes, path_);
}

void FileBoxWriter::sync_and_drop_cache() {
    if (fd_ < 0) {
        throw std::runtime_error("cannot sync a closed box writer: " + path_);
    }
    if (::fdatasync(fd_) != 0) {
        throw_errno("failed to sync", path_);
    }
    const std::uint64_t total =
        checked_add(kHeaderBytes, box_data_bytes(rows_, dims_), "box file size");
    advise_dontneed(fd_, 0, total, path_);
}

void initialize_box_file(const std::string& partial_path,
                         std::uint64_t rows, std::uint64_t dims) {
    if (dims == 0) {
        throw std::runtime_error("cannot create a zero-dimensional box store");
    }
    int fd = ::open(partial_path.c_str(),
                    O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        throw_errno("failed to create", partial_path);
    }
    try {
        const std::uint64_t total =
            checked_add(kHeaderBytes, box_data_bytes(rows, dims), "box file size");
        const off_t total_offset = checked_offset(total, "box file size");
        int status = ::posix_fallocate(fd, 0, total_offset);
        if (status != 0 && status != EOPNOTSUPP && status != ENOSYS) {
            errno = status;
            throw_errno("failed to allocate", partial_path);
        }
        if (::ftruncate(fd, total_offset) != 0) {
            throw_errno("failed to size", partial_path);
        }
        pwrite_all(fd, kMagic.data(), kMagic.size(), 0, partial_path);
        pwrite_all(fd, &rows, sizeof(rows), kRowsOffset, partial_path);
        pwrite_all(fd, &dims, sizeof(dims), kDimsOffset, partial_path);
        if (::fsync(fd) != 0) {
            throw_errno("failed to sync", partial_path);
        }
        ::close(fd);
    } catch (...) {
        ::close(fd);
        ::unlink(partial_path.c_str());
        throw;
    }
}

void finalize_box_file(const std::string& partial_path,
                       const std::string& final_path) {
    rename_noreplace(partial_path, final_path);
    sync_parent_directory(final_path);
}

std::uint64_t available_host_memory_bytes() {
    std::uint64_t available = 0;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            available = checked_mul(value, 1024, "MemAvailable bytes");
            break;
        }
    }
    if (available == 0) {
        const long pages = ::sysconf(_SC_AVPHYS_PAGES);
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0) {
            available = checked_mul(static_cast<std::uint64_t>(pages),
                                    static_cast<std::uint64_t>(page_size),
                                    "available host bytes");
        }
    }

    const std::filesystem::path cgroup =
        current_cgroup_v2_directory();
    bool finite_limit = false;
    const std::uint64_t limit =
        read_uint_file(cgroup / "memory.max", finite_limit);
    bool finite_high = false;
    const std::uint64_t high =
        read_uint_file(cgroup / "memory.high", finite_high);
    bool finite_current = false;
    const std::uint64_t current =
        read_uint_file(cgroup / "memory.current", finite_current);
    bool applied_cgroup_limit = false;
    if (finite_current && (finite_limit || finite_high)) {
        std::uint64_t effective_limit =
            std::numeric_limits<std::uint64_t>::max();
        if (finite_limit) {
            effective_limit = std::min(effective_limit, limit);
        }
        if (finite_high) {
            effective_limit = std::min(effective_limit, high);
        }
        const std::uint64_t remaining =
            current < effective_limit ? effective_limit - current : 0;
        available = available == 0 ? remaining : std::min(available, remaining);
        applied_cgroup_limit = true;
    }

    if (!applied_cgroup_limit) {
        const std::filesystem::path cgroup_v1 =
            current_cgroup_v1_memory_directory();
        bool finite_v1_limit = false;
        const std::uint64_t v1_limit = read_uint_file(
            cgroup_v1 / "memory.limit_in_bytes", finite_v1_limit);
        bool finite_v1_usage = false;
        const std::uint64_t v1_usage = read_uint_file(
            cgroup_v1 / "memory.usage_in_bytes", finite_v1_usage);
        if (finite_v1_limit && finite_v1_usage) {
            const std::uint64_t remaining =
                v1_usage < v1_limit ? v1_limit - v1_usage : 0;
            available = available == 0 ? remaining : std::min(available, remaining);
        }
    }
    return available;
}

std::uint64_t available_filesystem_bytes(const std::string& directory) {
    struct statvfs info {};
    if (::statvfs(directory.c_str(), &info) != 0) {
        throw_errno("failed to inspect filesystem", directory);
    }
    return checked_mul(static_cast<std::uint64_t>(info.f_bavail),
                       static_cast<std::uint64_t>(info.f_frsize),
                       "filesystem bytes");
}

}  // namespace tamubo::exactbo
