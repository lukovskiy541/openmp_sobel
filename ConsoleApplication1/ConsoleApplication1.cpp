#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define IDX(i, j, width) ((i) * (width) + (j))  
#define CLAMP(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x))) 


void skipCommentsAndWhitespace(FILE* fp) {
    int ch;
    char line_buffer[1024];
 
    while ((ch = fgetc(fp)) != EOF && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')) {}

    if (ch == '#') {
        fgets(line_buffer, sizeof(line_buffer), fp);
        skipCommentsAndWhitespace(fp);
    }
    
    else if (ch != EOF) {
        ungetc(ch, fp);
    }
}


unsigned char* loadImage(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    char magic[3];
    int max_val;
  
    if (fgets(magic, sizeof(magic), fp) == NULL || strncmp(magic, "P5", 2) != 0) {
        fprintf(stderr, "Error: Not a PGM P5 file.\n");
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

    if (fscanf(fp, "%d", &max_val) != 1 || max_val != 255) {
        fprintf(stderr, "Invalid max value.\n");
        fclose(fp);
        return NULL;
    }

    fgetc(fp); 


    size_t image_size = (size_t)(*width) * (*height);
    unsigned char* image_data = (unsigned char*)malloc(image_size);
    if (!image_data) {
        perror("malloc");
        fclose(fp);
        return NULL;
    }

   
    if (fread(image_data, 1, image_size, fp) != image_size) {
        fprintf(stderr, "Error reading image data.\n");
        free(image_data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return image_data;
}


void saveImage(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file");
        return;
    }


    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(image, 1, width * height, fp);
    fclose(fp);
}

void applySobel(unsigned char* input, unsigned char* output, int width, int height) {
    int sobelX[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
    int sobelY[3][3] = { {-1,-2,-1}, {0,0,0}, {1,2,1} };

#pragma omp parallel for collapse(2)  
    for (int i = 0; i < height; i++) {
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
    omp_set_num_threads(16);

    const char* input_filename = argc > 1 ? argv[1] : "input.pgm";
    const char* output_filename = argc > 2 ? argv[2] : "output.pgm";

    int width, height;
    
    unsigned char* image = loadImage(input_filename, &width, &height);
    if (!image) return 1;

    unsigned char* edges = (unsigned char*)malloc(width * height);
    if (!edges) {
        perror("malloc");
        free(image);
        return 1;
    }

    double start = omp_get_wtime(); 

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();
#pragma omp critical
       
        printf("Thread %d of %d is working\n", tid, total);
    }

  
    applySobel(image, edges, width, height);

    double end = omp_get_wtime();  

    printf("Processing time: %.5f seconds\n", end - start);  

  
    saveImage(output_filename, edges, width, height);

  
    free(image);
    free(edges);
    return 0;
}
