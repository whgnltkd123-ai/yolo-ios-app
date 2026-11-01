#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { float x, y, width, height, score; int32_t index; } OBTDetection;
typedef struct { float x, y, width, height, score; int32_t id, index; } OBTTrack;
void *obt_create(void);
void obt_destroy(void *handle);
int32_t obt_update(void *handle, const OBTDetection *input, int32_t count,
                   OBTTrack *output, int32_t capacity);
#ifdef __cplusplus
}
#endif
