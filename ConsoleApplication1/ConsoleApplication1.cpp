#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>


#define IDX(i, j, width) ((i) * (width) + (j))

#define CLAMP(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))


void skipCommentsAndWhitespace(FILE* fp) {
    int ch;
    char line_buffer[1024];

    while ((ch = fgetc(fp)) != EOF && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')) {
    }

    if (ch == '#') {
        fgets(line_buffer, sizeof(line_buffer), fp);
        skipCommentsAndWhitespace(fp);
    }
    else if (ch != EOF) {
        ungetc(ch, fp);
    }
}

unsigned char* loadImage(const char* filename, int* width, int* height) {
    FILE* fp = NULL;
    char magic[3];      
    int max_val = 0;
    unsigned char* image_data = NULL;
    size_t read_count;     

    fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    if (fgets(magic, sizeof(magic), fp) == NULL || strncmp(magic, "P5", 2) != 0) {
        fprintf(stderr, "Error: Not a PGM P5 file or error reading magic number.\n");
        fclose(fp);
        return NULL;
    }

    skipCommentsAndWhitespace(fp);
    if (fscanf(fp, "%d", width) != 1) {
        fprintf(stderr, "Error reading width.\n");
        fclose(fp);
        return NULL;
    }

    skipCommentsAndWhitespace(fp);
    if (fscanf(fp, "%d", height) != 1) {
        fprintf(stderr, "Error reading height.\n");
        fclose(fp);
        return NULL;
    }

    skipCommentsAndWhitespace(fp);
    if (fscanf(fp, "%d", &max_val) != 1) {
        fprintf(stderr, "Error reading max value.\n");
        fclose(fp);
        return NULL;
    }

    if (max_val != 255) {
        fprintf(stderr, "Warning: Max gray value is not 255 (%d), results might be unexpected.\n", max_val);
    }

    if (fgetc(fp) == EOF) {
        fprintf(stderr, "Error: Unexpected EOF after header.\n");
        fclose(fp);
        return NULL;
    }


    size_t image_size = (size_t)(*width) * (*height) * sizeof(unsigned char);
    image_data = (unsigned char*)malloc(image_size);
    if (!image_data) {
        perror("Error allocating memory for image");
        fclose(fp);
        return NULL;
    }

    read_count = fread(image_data, sizeof(unsigned char), (*width) * (*height), fp);
    if (read_count != (size_t)((*width) * (*height))) {
        fprintf(stderr, "Error reading pixel data: expected %zu bytes, got %zu\n",
            (size_t)(*width) * (*height), read_count);
        free(image_data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);

    printf("Loaded image: %s, %dx%d\n", filename, *width, *height);
    return image_data;
}

void saveImage(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = NULL;
    size_t write_count;

    fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file for writing");
        return;
    }

    fprintf(fp, "P5\n");   
    fprintf(fp, "%d %d\n", width, height);    
    fprintf(fp, "255\n");     

    write_count = fwrite(image, sizeof(unsigned char), (size_t)width * height, fp);
    if (write_count != (size_t)width * height) {
        fprintf(stderr, "Error writing pixel data: attempted %zu bytes, wrote %zu\n",
            (size_t)width * height, write_count);
    }

    if (fclose(fp) != 0) {
        perror("Error closing file after writing");
    }
    else {
        printf("Saved image: %s, %dx%d\n", filename, width, height);
    }
}

void applySobelToChunk(unsigned char* input, unsigned char* output,
    int width, int height, int startRow, int endRow) {

    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < width; j++) {
            int gradX = 0;
            int gradY = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int row = i + ki;
                    int col = j + kj;

                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        int pixel = input[IDX(row, col, width)];
                        gradX += pixel * sobelX[ki + 1][kj + 1];
                        gradY += pixel * sobelY[ki + 1][kj + 1];
                    }
                }
            }

            int magnitude = (int)sqrt(gradX * gradX + gradY * gradY);
            output[IDX(i, j, width)] = CLAMP(magnitude);
        }
    }
}

int main(int argc, char* argv[]) {
    int numprocs, MyID;
    int width, height;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    char (*all_proc_names)[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    int proc = 0;
    unsigned char* image = NULL;
    unsigned char* edges = NULL;
    unsigned char* local_result = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    MPI_Get_processor_name(processor_name, &namelen);

    all_proc_names = (char(*)[MPI_MAX_PROCESSOR_NAME]) malloc(numprocs * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
        all_proc_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (MyID == 0) {
        for (proc = 0; proc < numprocs; ++proc)
            printf("Process %d on %s\n", proc, all_proc_names[proc]);

        const char* input_filename = argc > 1 ? argv[1] : "input.pgm";
        image = loadImage(input_filename, &width, &height);
    }

    start_time = MPI_Wtime();

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (MyID != 0) {
        image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    }

    MPI_Bcast(image, width * height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int rows_per_proc = height / numprocs;
    int remainder = height % numprocs;

    int start_row = MyID * rows_per_proc + (MyID < remainder ? MyID : remainder);
    int end_row = start_row + rows_per_proc + (MyID < remainder ? 1 : 0);

    if (MyID == 0) {
        printf("Total image size: %dx%d\n", width, height);
        printf("Rows per process: ~%d\n", rows_per_proc);
    }

    printf("Process %d processing rows %d to %d\n", MyID, start_row, end_row - 1);

    local_result = (unsigned char*)malloc(width * (end_row - start_row) * sizeof(unsigned char));

    if (MyID == 0) {
        edges = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    }

    unsigned char* temp_output = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    applySobelToChunk(image, temp_output, width, height, start_row, end_row);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            local_result[IDX(i - start_row, j, width)] = temp_output[IDX(i, j, width)];
        }
    }

    free(temp_output);

    int* recvcounts = NULL;
    int* displs = NULL;

    if (MyID == 0) {
        recvcounts = (int*)malloc(numprocs * sizeof(int));
        displs = (int*)malloc(numprocs * sizeof(int));

        int offset = 0;
        for (int i = 0; i < numprocs; i++) {
            int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = proc_rows * width;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(local_result, (end_row - start_row) * width, MPI_UNSIGNED_CHAR,
        edges, recvcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (MyID == 0) {
        printf("Processing time: %.5f seconds\n", end_time - start_time);

        const char* output_filename = argc > 2 ? argv[2] : "output.pgm";
        saveImage(output_filename, edges, width, height);

        free(recvcounts);
        free(displs);
        free(edges);
    }

    free(image);
    free(local_result);
    free(all_proc_names);

    MPI_Finalize();
    return 0;
}