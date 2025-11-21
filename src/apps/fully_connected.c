#include "ipu/ipu.h"
#include "xmem/xmem.h"
#include "logging/logger.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_LAYER_FEATURES_NUM 600
#define OUTPUT_LAYER_FEATURES_NUM 300

#define FEATURES_BASE_ADDR 0
#define WEIGHTS_BASE_ADDR XMEM__ALIGN_ADDR(FEATURES_BASE_ADDR + INPUT_LAYER_FEATURES_NUM)
#define RESULTS_BASE_ADDR XMEM__ALIGN_ADDR(WEIGHTS_BASE_ADDR + (INPUT_LAYER_FEATURES_NUM * OUTPUT_LAYER_FEATURES_NUM))

#define INPUT_LAYER_WORDS XMEM__WORDS_NEEDED_FOR_BYTES(INPUT_LAYER_FEATURES_NUM)
#define OUTPUT_LAYER_WORDS XMEM__WORDS_NEEDED_FOR_BYTES(OUTPUT_LAYER_FEATURES_NUM)

#define IPU_REG_RES_INDEX_RQ 1
#define IPU_REG_FEATURE_INDEX_R 0
#define IPU_REG_WEIGHT_INDEX_R 1 

int main()
{
    logger__set_level(LOG_LEVEL_INFO);

    LOG_INFO("Initializing IPU...");
    ipu__obj_t *ipu = ipu__init_ipu();

    uint8_t *features = malloc(INPUT_LAYER_FEATURES_NUM * sizeof(uint8_t));
    uint8_t (*weights)[OUTPUT_LAYER_FEATURES_NUM] = malloc(INPUT_LAYER_FEATURES_NUM * OUTPUT_LAYER_FEATURES_NUM * sizeof(uint8_t));
    memset(features, 1, INPUT_LAYER_FEATURES_NUM * sizeof(uint8_t));
    memset(weights, 1, INPUT_LAYER_FEATURES_NUM * OUTPUT_LAYER_FEATURES_NUM * sizeof(uint8_t));

    LOG_INFO("Loading data to XMEM...");
    xmem__load_array_to(ipu->xmem, (uint8_t *)features, INPUT_LAYER_FEATURES_NUM, 0);
    LOG_INFO("Weights base address: %d", WEIGHTS_BASE_ADDR);
    xmem__load_matrix_to(ipu->xmem, (uint8_t *)weights, INPUT_LAYER_FEATURES_NUM, OUTPUT_LAYER_FEATURES_NUM, INPUT_LAYER_FEATURES_NUM);

    for (int i = 0; i < OUTPUT_LAYER_WORDS; i++)
    {
        LOG_INFO("Processing output layer word %d/%d...", i + 1, OUTPUT_LAYER_WORDS);
        ipu__clear_rq_reg(ipu, IPU_REG_RES_INDEX_RQ);

        for (int j = 0; j < INPUT_LAYER_FEATURES_NUM; j++)
        {
            LOG_DEBUG("  MAC input feature %d/%d...", j + 1, INPUT_LAYER_FEATURES_NUM);
            if (j % IPU__R_REG_SIZE_BYTES == 0)
            {
                ipu__load_r_reg(ipu, IPU_REG_FEATURE_INDEX_R, FEATURES_BASE_ADDR + (j * XMEM__XMEM_WIDTH_BYTES));
            }

            uint32_t offset = j * XMEM__ALIGN_ADDR(OUTPUT_LAYER_FEATURES_NUM) + (i * XMEM__XMEM_WIDTH_BYTES);
            ipu__load_r_reg(ipu, IPU_REG_WEIGHT_INDEX_R, WEIGHTS_BASE_ADDR + offset);

            ipu__mac_element_vector(ipu,
                                  IPU_REG_RES_INDEX_RQ,
                                  IPU_REG_WEIGHT_INDEX_R,
                                  IPU_REG_FEATURE_INDEX_R,
                                  j % IPU__R_REG_SIZE_BYTES,
                                  IPU__DATA_TYPE_INT8); 
        }
    }

    return 0;
}
