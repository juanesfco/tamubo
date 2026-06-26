#include <cstdio>
#include <mpi.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME] = {};
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    int cpu = -1;
#ifdef __linux__
    cpu = sched_getcpu();
#endif

    std::printf(
        "mpi_smoke: rank %d of %d on %s pid=%d cpu=%d\n", 
        rank, 
        size, 
        processor_name, 
        getpid(), 
        cpu
    );

    MPI_Finalize();
    return 0;
}
