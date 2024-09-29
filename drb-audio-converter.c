#include "drb-audio-converter.h"

/* -------------------------------------------------------------------------- */

#pragma mark - Includes

#if defined(__x86_64__)
#  if defined(__SSE4_1__)
#    define DRB_SIMD_SSE
#  endif
#elif defined(__aarch64__)
#  if defined(__ARM_NEON__)
#    define DRB_SIMD_NEON
#  endif
#endif

#if defined(DRB_SIMD_SSE)
#  include <x86intrin.h> // For `__m128`, `_mm_add_ps`, `_mm_mul_ps`, ...
#elif defined(DRB_SIMD_NEON)
#  include <arm_neon.h> // For `float32x4_t`, `vaddq_f32`, `vmulq_f32`, ...
#endif

#if defined(DEBUG)
#  include <assert.h> // For `assert`.
#  define ASSERT assert
#else
#  define ASSERT(...)
#endif

#include <string.h> // For `memcpy` and `memset`.

/* -------------------------------------------------------------------------- */

#pragma mark - Print Conversion

#if 0 // Enable to print the conversion into the console during construction.
#  define PRINT_CONVERSION
#endif

/* -------------------------------------------------------------------------- */

_Static_assert(sizeof(int) >= 4, "");

/* -------------------------------------------------------------------------- */

#pragma mark - Min and Max

static inline int min (int const x, int const y) { return x < y ? x : y; }
static inline int max (int const x, int const y) { return x > y ? x : y; }

/* -------------------------------------------------------------------------- */

#pragma mark - Bump Allocator

enum { cache_line_size = 64 };

static long cache_align (long const size)
{
    return (size + cache_line_size - 1) & -cache_line_size;
}

typedef struct Bump_Allocator
{
    unsigned char * memory;
    long offset;
}
Bump_Allocator;

static void bump_allocator_construct
    (
        Bump_Allocator * const allocator,
        void * const memory
    )
{
    allocator->memory = memory;
    allocator->offset = 0;
}

static void * alloc (Bump_Allocator * const allocator, long const size)
{
    ASSERT(size >= 0);

    long const offset = allocator->offset;

    allocator->offset += cache_align(size);

    return allocator->memory ? allocator->memory + offset : 0;
}

/* -------------------------------------------------------------------------- */

#pragma mark - Format, Processor, and Pass

enum
{
    channel_count_1 = 1,
    channel_count_2 = 2,
    channel_count_3 = 3,
    channel_count_4 = 4,
    channel_count_5 = 5,
    channel_count_6 = 6,
    channel_count_7 = 7,
    channel_count_8 = 8
};

enum
{
    layout_interleaved = 1,
    layout_deinterleaved = 2
};

typedef struct Drb_Audio_Converter_Format
{
    int sampling_rate;
    int channel_count;
    int layout;
    int block_size;
    int max_block_count;
}
Format;

typedef struct Processor
{
    int (* pushed_target_frame_count)
        (
            void const * state,
            int source_frame_count,
            double * latency
        );

    int (* pulled_source_frame_count)
        (
            void const * state,
            int target_frame_count,
            double * latency
        );

    int /* <- pushed target frame count */ (* push)
        (
            void * state,
            float * restrict source_samples,
            float * restrict target_samples,
            int source_frame_count
        );

    int /* <- pulled source frame count */ (* pull)
        (
            void * state,
            float * restrict source_samples,
            float * restrict target_samples,
            int target_frame_count
        );

    void * state;
}
Processor;

typedef struct Pass
{
    Processor (* create_processor)
        (
            void const * configuration,
            Bump_Allocator * allocator,
            Format const * source_format,
            Format const * target_format
        );

    void (* restrain_formats)
        (
            void const * configuration,
            Format * source_format,
            Format * target_format
        );

    void const * configuration;
}
Pass;

/* -------------------------------------------------------------------------- */

#pragma mark - Windows

typedef struct Window { float * samples; } Window;

enum
{
    window_8_size  =  8 * 2 * sizeof(float),
    window_16_size = 16 * 2 * sizeof(float),
    window_24_size = 24 * 2 * sizeof(float),
    window_32_size = 32 * 2 * sizeof(float),
    window_40_size = 40 * 2 * sizeof(float),
    window_48_size = 48 * 2 * sizeof(float)
};

static inline void window_8_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset + 0] = sample;
    window->samples[offset + 8] = sample;
}

static inline void window_16_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset +  0] = sample;
    window->samples[offset + 16] = sample;
}

static inline void window_24_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset +  0] = sample;
    window->samples[offset + 24] = sample;
}

static inline void window_32_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset +  0] = sample;
    window->samples[offset + 32] = sample;
}

static inline void window_40_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset +  0] = sample;
    window->samples[offset + 40] = sample;
}

static inline void window_48_write
    (
        Window * const window,
        float const sample,
        int const offset
    )
{
    window->samples[offset +  0] = sample;
    window->samples[offset + 48] = sample;
}

static inline int window_8_next_index (int const index)
{
    return (index + 1) & (8 - 1);
}

static inline int window_16_next_index (int const index)
{
    return (index + 1) & (16 - 1);
}

static inline int window_24_next_index (int const index)
{
    return index == (24 - 1) ? 0 : index + 1;
}

static inline int window_32_next_index (int const index)
{
    return (index + 1) & (32 - 1);
}

static inline int window_40_next_index (int const index)
{
    return index == (40 - 1) ? 0 : index + 1;
}

static inline int window_48_next_index (int const index)
{
    return index == (48 - 1) ? 0 : index + 1;
}

/* -------------------------------------------------------------------------- */

#pragma mark - Unify

static void unify (int * source_property, int * target_property)
{
    if (*source_property == 0 && *target_property > 0)
    {
        *source_property = *target_property;
    }
    else if (*source_property > 0 && *target_property == 0)
    {
        *target_property = *source_property;
    }

    ASSERT(*source_property == *target_property);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Convolution

#if defined(DRB_SIMD_SSE)

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];

    return acc;
}

static inline float convolve_4
    (
        float const kernel [static const 4],
        float const window [static const 4]
    )
{
    __m128 a = _mm_mul_ps(_mm_load_ps(kernel + 0), _mm_load_ps(window + 0));

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_8
    (
        float const kernel [static const 8],
        float const window [static const 8]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel + 0), _mm_load_ps(window + 0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel + 4), _mm_load_ps(window + 4));

    __m128 a = _mm_add_ps(t_0, t_1);

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_16
    (
        float const kernel [static const 16],
        float const window [static const 16]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel +  0), _mm_load_ps(window +  0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel +  4), _mm_load_ps(window +  4));
    __m128 t_2 = _mm_mul_ps(_mm_load_ps(kernel +  8), _mm_load_ps(window +  8));
    __m128 t_3 = _mm_mul_ps(_mm_load_ps(kernel + 12), _mm_load_ps(window + 12));

    __m128 a = _mm_add_ps(_mm_add_ps(t_0, t_1), _mm_add_ps(t_2, t_3));

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_24
    (
        float const kernel [static const 24],
        float const window [static const 24]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel +  0), _mm_load_ps(window +  0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel +  4), _mm_load_ps(window +  4));
    __m128 t_2 = _mm_mul_ps(_mm_load_ps(kernel +  8), _mm_load_ps(window +  8));
    __m128 t_3 = _mm_mul_ps(_mm_load_ps(kernel + 12), _mm_load_ps(window + 12));
    __m128 t_4 = _mm_mul_ps(_mm_load_ps(kernel + 16), _mm_load_ps(window + 16));
    __m128 t_5 = _mm_mul_ps(_mm_load_ps(kernel + 20), _mm_load_ps(window + 20));

    __m128 const a_0 = _mm_add_ps(_mm_add_ps(t_0, t_1), t_2);
    __m128 const a_1 = _mm_add_ps(_mm_add_ps(t_3, t_4), t_5);

    __m128 a = _mm_add_ps(a_0, a_1);

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_32
    (
        float const kernel [static const 32],
        float const window [static const 32]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel +  0), _mm_load_ps(window +  0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel +  4), _mm_load_ps(window +  4));
    __m128 t_2 = _mm_mul_ps(_mm_load_ps(kernel +  8), _mm_load_ps(window +  8));
    __m128 t_3 = _mm_mul_ps(_mm_load_ps(kernel + 12), _mm_load_ps(window + 12));
    __m128 t_4 = _mm_mul_ps(_mm_load_ps(kernel + 16), _mm_load_ps(window + 16));
    __m128 t_5 = _mm_mul_ps(_mm_load_ps(kernel + 20), _mm_load_ps(window + 20));
    __m128 t_6 = _mm_mul_ps(_mm_load_ps(kernel + 24), _mm_load_ps(window + 24));
    __m128 t_7 = _mm_mul_ps(_mm_load_ps(kernel + 28), _mm_load_ps(window + 28));

    __m128 const a_0 = _mm_add_ps(_mm_add_ps(t_0, t_1), _mm_add_ps(t_2, t_3));
    __m128 const a_1 = _mm_add_ps(_mm_add_ps(t_4, t_5), _mm_add_ps(t_6, t_7));

    __m128 a = _mm_add_ps(a_0, a_1);

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_40
    (
        float const kernel [static const 40],
        float const window [static const 40]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel +  0), _mm_load_ps(window +  0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel +  4), _mm_load_ps(window +  4));
    __m128 t_2 = _mm_mul_ps(_mm_load_ps(kernel +  8), _mm_load_ps(window +  8));
    __m128 t_3 = _mm_mul_ps(_mm_load_ps(kernel + 12), _mm_load_ps(window + 12));
    __m128 t_4 = _mm_mul_ps(_mm_load_ps(kernel + 16), _mm_load_ps(window + 16));
    __m128 t_5 = _mm_mul_ps(_mm_load_ps(kernel + 20), _mm_load_ps(window + 20));
    __m128 t_6 = _mm_mul_ps(_mm_load_ps(kernel + 24), _mm_load_ps(window + 24));
    __m128 t_7 = _mm_mul_ps(_mm_load_ps(kernel + 28), _mm_load_ps(window + 28));
    __m128 t_8 = _mm_mul_ps(_mm_load_ps(kernel + 32), _mm_load_ps(window + 32));
    __m128 t_9 = _mm_mul_ps(_mm_load_ps(kernel + 36), _mm_load_ps(window + 36));

    __m128 const a_0 = _mm_add_ps(_mm_add_ps(t_0, t_1), _mm_add_ps(t_2,  t_3));
    __m128 const a_1 = _mm_add_ps(_mm_add_ps(t_4, t_5), _mm_add_ps(t_6,  t_7));
    __m128 const a_2 = _mm_add_ps(a_0, t_8);
    __m128 const a_3 = _mm_add_ps(a_1, t_9);

    __m128 a = _mm_add_ps(a_2, a_3);

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

static inline float convolve_48
    (
        float const kernel [static const 48],
        float const window [static const 48]
    )
{
    __m128 t_0 = _mm_mul_ps(_mm_load_ps(kernel +  0), _mm_load_ps(window +  0));
    __m128 t_1 = _mm_mul_ps(_mm_load_ps(kernel +  4), _mm_load_ps(window +  4));
    __m128 t_2 = _mm_mul_ps(_mm_load_ps(kernel +  8), _mm_load_ps(window +  8));
    __m128 t_3 = _mm_mul_ps(_mm_load_ps(kernel + 12), _mm_load_ps(window + 12));
    __m128 t_4 = _mm_mul_ps(_mm_load_ps(kernel + 16), _mm_load_ps(window + 16));
    __m128 t_5 = _mm_mul_ps(_mm_load_ps(kernel + 20), _mm_load_ps(window + 20));
    __m128 t_6 = _mm_mul_ps(_mm_load_ps(kernel + 24), _mm_load_ps(window + 24));
    __m128 t_7 = _mm_mul_ps(_mm_load_ps(kernel + 28), _mm_load_ps(window + 28));
    __m128 t_8 = _mm_mul_ps(_mm_load_ps(kernel + 32), _mm_load_ps(window + 32));
    __m128 t_9 = _mm_mul_ps(_mm_load_ps(kernel + 36), _mm_load_ps(window + 36));
    __m128 t_10= _mm_mul_ps(_mm_load_ps(kernel + 40), _mm_load_ps(window + 40));
    __m128 t_11= _mm_mul_ps(_mm_load_ps(kernel + 44), _mm_load_ps(window + 44));

    __m128 const a_0 = _mm_add_ps(_mm_add_ps(t_0, t_1), _mm_add_ps(t_2,  t_3));
    __m128 const a_1 = _mm_add_ps(_mm_add_ps(t_4, t_5), _mm_add_ps(t_6,  t_7));
    __m128 const a_2 = _mm_add_ps(_mm_add_ps(t_8, t_9), _mm_add_ps(t_10, t_11));

    __m128 a = _mm_add_ps(_mm_add_ps(a_0, a_1), a_2);

    a = _mm_hadd_ps(a, a);

    return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

#elif defined(DRB_SIMD_NEON)

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    return vaddv_f32(vmul_f32(vld1_f32(kernel), vld1_f32(window)));
}

static inline float convolve_4
    (
        float const kernel [static const 4],
        float const window [static const 4]
    )
{
    return vaddvq_f32(vmulq_f32(vld1q_f32(kernel), vld1q_f32(window)));
}

static inline float convolve_8
    (
        float const kernel [static const 8],
        float const window [static const 8]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel + 0), vld1q_f32(window + 0));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 4), vld1q_f32(window + 4));

    return vaddvq_f32(a_0);
}

static inline float convolve_16
    (
        float const kernel [static const 16],
        float const window [static const 16]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel + 0), vld1q_f32(window + 0));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  4), vld1q_f32(window +  4));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  8), vld1q_f32(window +  8));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 12), vld1q_f32(window + 12));

    return vaddvq_f32(a_0);
}

static inline float convolve_24
    (
        float const kernel [static const 24],
        float const window [static const 24]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel +  0), vld1q_f32(window +  0));
    float32x4_t a_1 = vmulq_f32(vld1q_f32(kernel + 12), vld1q_f32(window + 12));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  4), vld1q_f32(window +  4));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  8), vld1q_f32(window +  8));

    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 16), vld1q_f32(window + 16));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 20), vld1q_f32(window + 20));

    return vaddvq_f32(vaddq_f32(a_0, a_1));
}

static inline float convolve_32
    (
        float const kernel [static const 32],
        float const window [static const 32]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel +  0), vld1q_f32(window +  0));
    float32x4_t a_1 = vmulq_f32(vld1q_f32(kernel + 16), vld1q_f32(window + 16));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  4), vld1q_f32(window +  4));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  8), vld1q_f32(window +  8));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 12), vld1q_f32(window + 12));

    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 20), vld1q_f32(window + 20));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 24), vld1q_f32(window + 24));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 28), vld1q_f32(window + 28));

    return vaddvq_f32(vaddq_f32(a_0, a_1));
}

static inline float convolve_40
    (
        float const kernel [static const 40],
        float const window [static const 40]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel +  0), vld1q_f32(window +  0));
    float32x4_t a_1 = vmulq_f32(vld1q_f32(kernel + 20), vld1q_f32(window + 20));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  4), vld1q_f32(window +  4));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  8), vld1q_f32(window +  8));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 12), vld1q_f32(window + 12));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 16), vld1q_f32(window + 16));

    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 24), vld1q_f32(window + 24));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 28), vld1q_f32(window + 28));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 32), vld1q_f32(window + 32));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 36), vld1q_f32(window + 36));

    return vaddvq_f32(vaddq_f32(a_0, a_1));
}

static inline float convolve_48
    (
        float const kernel [static const 48],
        float const window [static const 48]
    )
{
    float32x4_t a_0 = vmulq_f32(vld1q_f32(kernel +  0), vld1q_f32(window +  0));
    float32x4_t a_1 = vmulq_f32(vld1q_f32(kernel + 16), vld1q_f32(window + 16));
    float32x4_t a_2 = vmulq_f32(vld1q_f32(kernel + 32), vld1q_f32(window + 32));

    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  4), vld1q_f32(window +  4));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel +  8), vld1q_f32(window +  8));
    a_0 = vfmaq_f32(a_0, vld1q_f32(kernel + 12), vld1q_f32(window + 12));

    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 20), vld1q_f32(window + 20));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 24), vld1q_f32(window + 24));
    a_1 = vfmaq_f32(a_1, vld1q_f32(kernel + 28), vld1q_f32(window + 28));

    a_2 = vfmaq_f32(a_2, vld1q_f32(kernel + 36), vld1q_f32(window + 36));
    a_2 = vfmaq_f32(a_2, vld1q_f32(kernel + 40), vld1q_f32(window + 40));
    a_2 = vfmaq_f32(a_2, vld1q_f32(kernel + 44), vld1q_f32(window + 44));

    return vaddvq_f32(vaddq_f32(vaddq_f32(a_0, a_1), a_2));
}

#else // No SIMD

static inline float convolve_2
    (
        float const kernel [static const 2],
        float const window [static const 2]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];

    return acc;
}

static inline float convolve_4
    (
        float const kernel [static const 4],
        float const window [static const 4]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];

    return acc;
}

static inline float convolve_8
    (
        float const kernel [static const 8],
        float const window [static const 8]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];

    return acc;
}

static inline float convolve_16
    (
        float const kernel [static const 16],
        float const window [static const 16]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];
    acc += kernel[ 8] * window[ 8]; acc += kernel[ 9] * window[ 9];
    acc += kernel[10] * window[10]; acc += kernel[11] * window[11];
    acc += kernel[12] * window[12]; acc += kernel[13] * window[13];
    acc += kernel[14] * window[14]; acc += kernel[15] * window[15];

    return acc;
}

static inline float convolve_24
    (
        float const kernel [static const 24],
        float const window [static const 24]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];
    acc += kernel[ 8] * window[ 8]; acc += kernel[ 9] * window[ 9];
    acc += kernel[10] * window[10]; acc += kernel[11] * window[11];
    acc += kernel[12] * window[12]; acc += kernel[13] * window[13];
    acc += kernel[14] * window[14]; acc += kernel[15] * window[15];
    acc += kernel[16] * window[16]; acc += kernel[17] * window[17];
    acc += kernel[18] * window[18]; acc += kernel[19] * window[19];
    acc += kernel[20] * window[20]; acc += kernel[21] * window[21];
    acc += kernel[22] * window[22]; acc += kernel[23] * window[23];

    return acc;
}

static inline float convolve_32
    (
        float const kernel [static const 32],
        float const window [static const 32]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];
    acc += kernel[ 8] * window[ 8]; acc += kernel[ 9] * window[ 9];
    acc += kernel[10] * window[10]; acc += kernel[11] * window[11];
    acc += kernel[12] * window[12]; acc += kernel[13] * window[13];
    acc += kernel[14] * window[14]; acc += kernel[15] * window[15];
    acc += kernel[16] * window[16]; acc += kernel[17] * window[17];
    acc += kernel[18] * window[18]; acc += kernel[19] * window[19];
    acc += kernel[20] * window[20]; acc += kernel[21] * window[21];
    acc += kernel[22] * window[22]; acc += kernel[23] * window[23];
    acc += kernel[24] * window[24]; acc += kernel[25] * window[25];
    acc += kernel[26] * window[26]; acc += kernel[27] * window[27];
    acc += kernel[28] * window[28]; acc += kernel[29] * window[29];
    acc += kernel[30] * window[30]; acc += kernel[31] * window[31];

    return acc;
}

static inline float convolve_40
    (
        float const kernel [static const 40],
        float const window [static const 40]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];
    acc += kernel[ 8] * window[ 8]; acc += kernel[ 9] * window[ 9];
    acc += kernel[10] * window[10]; acc += kernel[11] * window[11];
    acc += kernel[12] * window[12]; acc += kernel[13] * window[13];
    acc += kernel[14] * window[14]; acc += kernel[15] * window[15];
    acc += kernel[16] * window[16]; acc += kernel[17] * window[17];
    acc += kernel[18] * window[18]; acc += kernel[19] * window[19];
    acc += kernel[20] * window[20]; acc += kernel[21] * window[21];
    acc += kernel[22] * window[22]; acc += kernel[23] * window[23];
    acc += kernel[24] * window[24]; acc += kernel[25] * window[25];
    acc += kernel[26] * window[26]; acc += kernel[27] * window[27];
    acc += kernel[28] * window[28]; acc += kernel[29] * window[29];
    acc += kernel[30] * window[30]; acc += kernel[31] * window[31];
    acc += kernel[32] * window[32]; acc += kernel[33] * window[33];
    acc += kernel[34] * window[34]; acc += kernel[35] * window[35];
    acc += kernel[36] * window[36]; acc += kernel[37] * window[37];
    acc += kernel[38] * window[38]; acc += kernel[39] * window[39];

    return acc;
}

static inline float convolve_48
    (
        float const kernel [static const 48],
        float const window [static const 48]
    )
{
    float acc = 0.0f;

    acc += kernel[ 0] * window[ 0]; acc += kernel[ 1] * window[ 1];
    acc += kernel[ 2] * window[ 2]; acc += kernel[ 3] * window[ 3];
    acc += kernel[ 4] * window[ 4]; acc += kernel[ 5] * window[ 5];
    acc += kernel[ 6] * window[ 6]; acc += kernel[ 7] * window[ 7];
    acc += kernel[ 8] * window[ 8]; acc += kernel[ 9] * window[ 9];
    acc += kernel[10] * window[10]; acc += kernel[11] * window[11];
    acc += kernel[12] * window[12]; acc += kernel[13] * window[13];
    acc += kernel[14] * window[14]; acc += kernel[15] * window[15];
    acc += kernel[16] * window[16]; acc += kernel[17] * window[17];
    acc += kernel[18] * window[18]; acc += kernel[19] * window[19];
    acc += kernel[20] * window[20]; acc += kernel[21] * window[21];
    acc += kernel[22] * window[22]; acc += kernel[23] * window[23];
    acc += kernel[24] * window[24]; acc += kernel[25] * window[25];
    acc += kernel[26] * window[26]; acc += kernel[27] * window[27];
    acc += kernel[28] * window[28]; acc += kernel[29] * window[29];
    acc += kernel[30] * window[30]; acc += kernel[31] * window[31];
    acc += kernel[32] * window[32]; acc += kernel[33] * window[33];
    acc += kernel[34] * window[34]; acc += kernel[35] * window[35];
    acc += kernel[36] * window[36]; acc += kernel[37] * window[37];
    acc += kernel[38] * window[38]; acc += kernel[39] * window[39];
    acc += kernel[40] * window[40]; acc += kernel[41] * window[41];
    acc += kernel[42] * window[42]; acc += kernel[43] * window[43];
    acc += kernel[44] * window[44]; acc += kernel[45] * window[45];
    acc += kernel[46] * window[46]; acc += kernel[47] * window[47];

    return acc;
}

#endif // SIMD

/* -------------------------------------------------------------------------- */

#pragma mark - Interpolation

enum
{
    kernel_count_sinc_8_linear  = 64,
    kernel_count_sinc_16_linear = 64,
    kernel_count_sinc_24_linear = 64,
    kernel_count_sinc_32_linear = 64,
    kernel_count_sinc_8_cubic   = 32,
    kernel_count_sinc_16_cubic  = 32,
    kernel_count_sinc_24_cubic  = 32,
    kernel_count_sinc_32_cubic  = 32,
};

static float const kernels_sinc_8_linear  [kernel_count_sinc_8_linear ][ 8];
static float const kernels_sinc_16_linear [kernel_count_sinc_16_linear][16];
static float const kernels_sinc_24_linear [kernel_count_sinc_16_linear][24];
static float const kernels_sinc_32_linear [kernel_count_sinc_32_linear][32];
static float const kernels_sinc_8_cubic   [kernel_count_sinc_16_cubic ][ 8];
static float const kernels_sinc_16_cubic  [kernel_count_sinc_16_cubic ][16];
static float const kernels_sinc_24_cubic  [kernel_count_sinc_16_cubic ][24];
static float const kernels_sinc_32_cubic  [kernel_count_sinc_32_cubic ][32];

static inline float interpolate_linear
    (
        float const blend,
        float const window [static const 2]
    )
{
    // Uses linear interpolation to interpolate between two samples in a 2-point
    // window. Say the window is [`p0`, `p1`], `interpolate_linear` will blend
    // between `p0` and `p1` according to the value provided in the `blend`
    // parameter, which must be in the range [0; 1].

    static float const mat [2][2] =
    {
        { +1.0f,  0.0f },
        { -1.0f, +1.0f }
    };

    float const a_0 = convolve_2(mat[0], window);
    float const a_1 = convolve_2(mat[1], window);

    return a_1 * blend + a_0;
}

static inline float interpolate_cubic
    (
        float const blend,
        float const window [static const 4]
    )
{
    // Uses a Catmull–Rom spline to interpolate between the two middle samples
    // in a 4-sample window. Say the window is [`p0`, `p1`, `p2`, `p3`],
    // this function will blend between `p1` and `p2` according to the value
    // provided in the `blend` parameter, which must be in the range [0; 1].

    static float const mat [4][4] =
    {
        {         0.0f, 0.5f * +2.0f,         0.0f,         0.0f },
        { 0.5f * -1.0f,         0.0f, 0.5f * +1.0f,         0.0f },
        { 0.5f * +2.0f, 0.5f * -5.0f, 0.5f * +4.0f, 0.5f * -1.0f },
        { 0.5f * -1.0f, 0.5f * +3.0f, 0.5f * -3.0f, 0.5f * +1.0f }
    };

    float const a_0 = convolve_4(mat[0], window);
    float const a_1 = convolve_4(mat[1], window);
    float const a_2 = convolve_4(mat[2], window);
    float const a_3 = convolve_4(mat[3], window);

    return (((a_3 * blend + a_2) * blend) + a_1) * blend + a_0;
}

// Uses a bank of sincs to interpolate between the two middle samples in an 8,
// 16, 24, or 32-sample window. Say the window is [`p0`, `p1`, ..., `p7`, `p8`],
// this function will blend between `p3` and `p4` according to the value
// provided in the `blend` parameter, which must be in the range [0; 1].
#define INTERPOLATE_SINC_LINEAR(SIZE)                                          \
static inline float interpolate_sinc_##SIZE##_linear                           \
    (                                                                          \
        double const blend,                                                    \
        float const window [static const SIZE])                                \
{                                                                              \
    static double const scale = kernel_count_sinc_##SIZE##_linear - 1;         \
                                                                               \
    int const index = (int)(blend * scale);                                    \
                                                                               \
    ASSERT(index < kernel_count_sinc_##SIZE##_linear);                         \
                                                                               \
    _Alignas(8) float const points [2] =                                       \
    {                                                                          \
        convolve_##SIZE(kernels_sinc_##SIZE##_linear[index + 0], window),      \
        convolve_##SIZE(kernels_sinc_##SIZE##_linear[index + 1], window)       \
    };                                                                         \
                                                                               \
    double const sub_blend = blend * scale - (double)index;                    \
                                                                               \
    return interpolate_linear(sub_blend, points);                              \
}                                                                              \

INTERPOLATE_SINC_LINEAR( 8) // => `interpolate_sinc_8_linear`
INTERPOLATE_SINC_LINEAR(16) // => `interpolate_sinc_16_linear`
INTERPOLATE_SINC_LINEAR(24) // => `interpolate_sinc_24_linear`
INTERPOLATE_SINC_LINEAR(32) // => `interpolate_sinc_32_linear`

// Uses a bank of sincs to interpolate between the two middle samples in an 8,
// 16, 24, or 32-sample window. Say the window is [`p0`, `p1`, ..., `p7`, `p8`],
// this function will blend between `p3` and `p4` according to the value
// provided in the `blend` parameter, which must be in the range [0; 1].
#define INTERPOLATE_SINC_CUBIC(SIZE)                                           \
static inline float interpolate_sinc_##SIZE##_cubic                            \
    (                                                                          \
        double const blend,                                                    \
        float const window [static const SIZE])                                \
{                                                                              \
    static double const scale = kernel_count_sinc_##SIZE##_cubic - 3;          \
                                                                               \
    int const index = (int)(blend * scale);                                    \
                                                                               \
    ASSERT(index < kernel_count_sinc_##SIZE##_cubic);                          \
                                                                               \
    _Alignas(16) float const points [4] =                                      \
    {                                                                          \
        convolve_##SIZE(kernels_sinc_##SIZE##_cubic[index + 0], window),       \
        convolve_##SIZE(kernels_sinc_##SIZE##_cubic[index + 1], window),       \
        convolve_##SIZE(kernels_sinc_##SIZE##_cubic[index + 2], window),       \
        convolve_##SIZE(kernels_sinc_##SIZE##_cubic[index + 3], window)        \
    };                                                                         \
                                                                               \
    double const sub_blend = blend * scale - (double)index;                    \
                                                                               \
    return interpolate_cubic(sub_blend, points);                               \
}                                                                              \

INTERPOLATE_SINC_CUBIC( 8) // => `interpolate_sinc_8_linear`
INTERPOLATE_SINC_CUBIC(16) // => `interpolate_sinc_16_linear`
INTERPOLATE_SINC_CUBIC(24) // => `interpolate_sinc_24_linear`
INTERPOLATE_SINC_CUBIC(32) // => `interpolate_sinc_32_linear`

/* -------------------------------------------------------------------------- */

#pragma mark - Layout

static int interleave_2
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 2 + 0] = source_samples_0[frame];
        target_samples[frame * 2 + 1] = source_samples_1[frame];
    }

    return frame_count;
}

static int interleave_3
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 3 + 0] = source_samples_0[frame];
        target_samples[frame * 3 + 1] = source_samples_1[frame];
        target_samples[frame * 3 + 2] = source_samples_2[frame];
    }

    return frame_count;
}

static int interleave_4
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 4 + 0] = source_samples_0[frame];
        target_samples[frame * 4 + 1] = source_samples_1[frame];
        target_samples[frame * 4 + 2] = source_samples_2[frame];
        target_samples[frame * 4 + 3] = source_samples_3[frame];
    }

    return frame_count;
}

static int interleave_5
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 5 + 0] = source_samples_0[frame];
        target_samples[frame * 5 + 1] = source_samples_1[frame];
        target_samples[frame * 5 + 2] = source_samples_2[frame];
        target_samples[frame * 5 + 3] = source_samples_3[frame];
        target_samples[frame * 5 + 4] = source_samples_4[frame];
    }

    return frame_count;
}

static int interleave_6
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 6 + 0] = source_samples_0[frame];
        target_samples[frame * 6 + 1] = source_samples_1[frame];
        target_samples[frame * 6 + 2] = source_samples_2[frame];
        target_samples[frame * 6 + 3] = source_samples_3[frame];
        target_samples[frame * 6 + 4] = source_samples_4[frame];
        target_samples[frame * 6 + 5] = source_samples_5[frame];
    }

    return frame_count;
}

static int interleave_7
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;
    float * const source_samples_6 = source_samples + frame_count * 6;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 7 + 0] = source_samples_0[frame];
        target_samples[frame * 7 + 1] = source_samples_1[frame];
        target_samples[frame * 7 + 2] = source_samples_2[frame];
        target_samples[frame * 7 + 3] = source_samples_3[frame];
        target_samples[frame * 7 + 4] = source_samples_4[frame];
        target_samples[frame * 7 + 5] = source_samples_5[frame];
        target_samples[frame * 7 + 6] = source_samples_6[frame];
    }

    return frame_count;
}

static int interleave_8
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const source_samples_0 = source_samples + frame_count * 0;
    float * const source_samples_1 = source_samples + frame_count * 1;
    float * const source_samples_2 = source_samples + frame_count * 2;
    float * const source_samples_3 = source_samples + frame_count * 3;
    float * const source_samples_4 = source_samples + frame_count * 4;
    float * const source_samples_5 = source_samples + frame_count * 5;
    float * const source_samples_6 = source_samples + frame_count * 6;
    float * const source_samples_7 = source_samples + frame_count * 7;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples[frame * 8 + 0] = source_samples_0[frame];
        target_samples[frame * 8 + 1] = source_samples_1[frame];
        target_samples[frame * 8 + 2] = source_samples_2[frame];
        target_samples[frame * 8 + 3] = source_samples_3[frame];
        target_samples[frame * 8 + 4] = source_samples_4[frame];
        target_samples[frame * 8 + 5] = source_samples_5[frame];
        target_samples[frame * 8 + 6] = source_samples_6[frame];
        target_samples[frame * 8 + 7] = source_samples_7[frame];
    }

    return frame_count;
}

static int deinterleave_2
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 2 + 0];
        target_samples_1[frame] = source_samples[frame * 2 + 1];
    }

    return frame_count;
}

static int deinterleave_3
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 3 + 0];
        target_samples_1[frame] = source_samples[frame * 3 + 1];
        target_samples_2[frame] = source_samples[frame * 3 + 2];
    }

    return frame_count;
}

static int deinterleave_4
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 4 + 0];
        target_samples_1[frame] = source_samples[frame * 4 + 1];
        target_samples_2[frame] = source_samples[frame * 4 + 2];
        target_samples_3[frame] = source_samples[frame * 4 + 3];
    }

    return frame_count;
}

static int deinterleave_5
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 5 + 0];
        target_samples_1[frame] = source_samples[frame * 5 + 1];
        target_samples_2[frame] = source_samples[frame * 5 + 2];
        target_samples_3[frame] = source_samples[frame * 5 + 3];
        target_samples_4[frame] = source_samples[frame * 5 + 4];
    }

    return frame_count;
}

static int deinterleave_6
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 6 + 0];
        target_samples_1[frame] = source_samples[frame * 6 + 1];
        target_samples_2[frame] = source_samples[frame * 6 + 2];
        target_samples_3[frame] = source_samples[frame * 6 + 3];
        target_samples_4[frame] = source_samples[frame * 6 + 4];
        target_samples_5[frame] = source_samples[frame * 6 + 5];
    }

    return frame_count;
}

static int deinterleave_7
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;
    float * const target_samples_6 = target_samples + frame_count * 6;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 7 + 0];
        target_samples_1[frame] = source_samples[frame * 7 + 1];
        target_samples_2[frame] = source_samples[frame * 7 + 2];
        target_samples_3[frame] = source_samples[frame * 7 + 3];
        target_samples_4[frame] = source_samples[frame * 7 + 4];
        target_samples_5[frame] = source_samples[frame * 7 + 5];
        target_samples_6[frame] = source_samples[frame * 7 + 6];
    }

    return frame_count;
}

static int deinterleave_8
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const frame_count
    )
{
    (void)state;

    float * const target_samples_0 = target_samples + frame_count * 0;
    float * const target_samples_1 = target_samples + frame_count * 1;
    float * const target_samples_2 = target_samples + frame_count * 2;
    float * const target_samples_3 = target_samples + frame_count * 3;
    float * const target_samples_4 = target_samples + frame_count * 4;
    float * const target_samples_5 = target_samples + frame_count * 5;
    float * const target_samples_6 = target_samples + frame_count * 6;
    float * const target_samples_7 = target_samples + frame_count * 7;

    for (int frame = 0; frame < frame_count; frame++)
    {
        target_samples_0[frame] = source_samples[frame * 8 + 0];
        target_samples_1[frame] = source_samples[frame * 8 + 1];
        target_samples_2[frame] = source_samples[frame * 8 + 2];
        target_samples_3[frame] = source_samples[frame * 8 + 3];
        target_samples_4[frame] = source_samples[frame * 8 + 4];
        target_samples_5[frame] = source_samples[frame * 8 + 5];
        target_samples_6[frame] = source_samples[frame * 8 + 6];
        target_samples_7[frame] = source_samples[frame * 8 + 7];
    }

    return frame_count;
}

static Processor interleaver_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration, (void)allocator;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->channel_count >= 2);
    ASSERT(target_format->channel_count >= 2);
    ASSERT(source_format->layout == layout_deinterleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size == target_format->block_size);
    ASSERT(source_format->max_block_count == target_format->max_block_count);

    int (* callbacks [])(void *, float * restrict, float * restrict, int) =
    {
        [2] = interleave_2,
        [3] = interleave_3,
        [4] = interleave_4,
        [5] = interleave_5,
        [6] = interleave_6,
        [7] = interleave_7,
        [8] = interleave_8
    };

    return (Processor)
    {
        .push = callbacks[source_format->channel_count],
        .pull = callbacks[target_format->channel_count]
    };
}

static Processor deinterleaver_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration, (void)allocator;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->channel_count >= 2);
    ASSERT(target_format->channel_count >= 2);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_deinterleaved);
    ASSERT(source_format->block_size == target_format->block_size);
    ASSERT(source_format->max_block_count == target_format->max_block_count);

    int (* callbacks [])(void *, float * restrict, float * restrict, int) =
    {
        [2] = deinterleave_2,
        [3] = deinterleave_3,
        [4] = deinterleave_4,
        [5] = deinterleave_5,
        [6] = deinterleave_6,
        [7] = deinterleave_7,
        [8] = deinterleave_8
    };

    return (Processor)
    {
        .push = callbacks[source_format->channel_count],
        .pull = callbacks[target_format->channel_count]
    };
}

static void interleaver_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->layout == 0)
    {
        source_format->layout = layout_deinterleaved;
    }

    if (target_format->layout == 0)
    {
        target_format->layout = layout_interleaved;
    }

    ASSERT(source_format->layout == layout_deinterleaved);
    ASSERT(target_format->layout == layout_interleaved);

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->block_size, &target_format->block_size);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static void deinterleaver_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->layout == 0)
    {
        source_format->layout = layout_interleaved;
    }

    if (target_format->layout == 0)
    {
        target_format->layout = layout_deinterleaved;
    }

    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_deinterleaved);

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->block_size, &target_format->block_size);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static const Pass interleaver =
{
    .create_processor = interleaver_create_processor,
    .restrain_formats = interleaver_restrain_formats
};

static const Pass deinterleaver =
{
    .create_processor = deinterleaver_create_processor,
    .restrain_formats = deinterleaver_restrain_formats
};

/* -------------------------------------------------------------------------- */

#pragma mark - Sinc Resampling

static int const resampler_sinc_mode_8_linear  = 0;
static int const resampler_sinc_mode_16_linear = 1;
static int const resampler_sinc_mode_24_linear = 2;
static int const resampler_sinc_mode_32_linear = 3;
static int const resampler_sinc_mode_8_cubic   = 4;
static int const resampler_sinc_mode_16_cubic  = 5;
static int const resampler_sinc_mode_24_cubic  = 6;
static int const resampler_sinc_mode_32_cubic  = 7;

typedef struct Resampler_Sinc
{
    long long source_delta;
    long long target_delta;
    long long phase;
    double latency;
    double scale;
    int channel_count;
    int index;
    _Alignas(cache_line_size) Window windows [];
}
Resampler_Sinc;

static int resampler_sinc_target_frame_count
    (
        void const * const state,
        int const source_frame_count,
        double * const latency
    )
{
    Resampler_Sinc const * const resampler = state;

    long long count = resampler->target_delta - resampler->phase;

    count += resampler->source_delta * source_frame_count;
    count /= resampler->target_delta;

    *latency = (resampler->phase - 1) * resampler->scale - resampler->latency;

    return (int)count;
}

static int resampler_sinc_source_frame_count
    (
        void const * const state,
        int const target_frame_count,
        double * const latency
    )
{
    Resampler_Sinc const * const resampler = state;

    long long count = resampler->phase - resampler->target_delta;

    count += resampler->target_delta * target_frame_count;
    count /= resampler->source_delta;

    *latency = resampler->phase * resampler->scale - resampler->latency;

    return (int)count;
}

#define RESAMPLER_SINC_UPDATE_WINDOWS(SIZE)                                    \
static inline void resampler_sinc_update_windows_##SIZE                        \
    (                                                                          \
        Resampler_Sinc * const resampler,                                      \
        int const source_frame,                                                \
        float const * const source_samples                                     \
    )                                                                          \
{                                                                              \
    for (int channel = 0; channel < resampler->channel_count; channel++)       \
    {                                                                          \
        Window * const window = &resampler->windows[channel];                  \
                                                                               \
        int const index = resampler->channel_count * source_frame + channel;   \
                                                                               \
        window_##SIZE##_write(window, source_samples[index], resampler->index);\
    }                                                                          \
                                                                               \
    resampler->index = window_##SIZE##_next_index(resampler->index);           \
}                                                                              \

RESAMPLER_SINC_UPDATE_WINDOWS( 8) // => `resampler_sinc_update_windows_8`
RESAMPLER_SINC_UPDATE_WINDOWS(16) // => `resampler_sinc_update_windows_16`
RESAMPLER_SINC_UPDATE_WINDOWS(24) // => `resampler_sinc_update_windows_24`
RESAMPLER_SINC_UPDATE_WINDOWS(32) // => `resampler_sinc_update_windows_32`

#define RESAMPLE_SINC_FRAME(SIZE, INTERPOLATION)                               \
static inline void resample_sinc_frame_##SIZE##_##INTERPOLATION                \
    (                                                                          \
        Resampler_Sinc * const resampler,                                      \
        int const target_frame,                                                \
        float * const target_samples,                                          \
        double const blend                                                     \
    )                                                                          \
{                                                                              \
    for (int channel = 0; channel < resampler->channel_count; channel++)       \
    {                                                                          \
        Window const * const window = &resampler->windows[channel];            \
                                                                               \
        int const index = resampler->channel_count * target_frame + channel;   \
                                                                               \
        float const * const slice = window->samples + resampler->index;        \
                                                                               \
        target_samples[index] = interpolate_sinc_##SIZE##_##INTERPOLATION      \
        (                                                                      \
            blend,                                                             \
            slice                                                              \
        );                                                                     \
    }                                                                          \
}                                                                              \

RESAMPLE_SINC_FRAME( 8, linear) // => `resample_sinc_frame_8_linear`
RESAMPLE_SINC_FRAME(16, linear) // => `resample_sinc_frame_16_linear`
RESAMPLE_SINC_FRAME(24, linear) // => `resample_sinc_frame_24_linear`
RESAMPLE_SINC_FRAME(32, linear) // => `resample_sinc_frame_32_linear`
RESAMPLE_SINC_FRAME( 8, cubic ) // => `resample_sinc_frame_8_cubic`
RESAMPLE_SINC_FRAME(16, cubic ) // => `resample_sinc_frame_16_cubic`
RESAMPLE_SINC_FRAME(24, cubic ) // => `resample_sinc_frame_24_cubic`
RESAMPLE_SINC_FRAME(32, cubic ) // => `resample_sinc_frame_32_cubic`

#define RESAMPLER_SINC_PUSH(SIZE, INTERPOLATION)                               \
static int resampler_sinc_push_##SIZE##_##INTERPOLATION                        \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    Resampler_Sinc * const resampler = state;                                  \
                                                                               \
    int source_frame = 0, target_frame = 0;                                    \
                                                                               \
    while (source_frame < source_frame_count)                                  \
    {                                                                          \
        while (resampler->phase <= resampler->source_delta)                    \
        {                                                                      \
            double const blend = (resampler->phase - 1) * resampler->scale;    \
                                                                               \
            ASSERT(0.0f <= blend && blend < 1.0f);                             \
                                                                               \
            resample_sinc_frame_##SIZE##_##INTERPOLATION                       \
            (                                                                  \
                resampler,                                                     \
                target_frame,                                                  \
                target_samples,                                                \
                blend                                                          \
            );                                                                 \
                                                                               \
            resampler->phase += resampler->target_delta;                       \
                                                                               \
            target_frame++;                                                    \
        }                                                                      \
                                                                               \
        resampler_sinc_update_windows_##SIZE                                   \
        (                                                                      \
            resampler,                                                         \
            source_frame,                                                      \
            source_samples                                                     \
        );                                                                     \
                                                                               \
        resampler->phase -= resampler->source_delta;                           \
                                                                               \
        source_frame++;                                                        \
    }                                                                          \
                                                                               \
    ASSERT(resampler->phase >= 0);                                             \
                                                                               \
    return target_frame;                                                       \
}                                                                              \

RESAMPLER_SINC_PUSH( 8, linear) // => `resampler_sinc_push_8_linear`
RESAMPLER_SINC_PUSH(16, linear) // => `resampler_sinc_push_16_linear`
RESAMPLER_SINC_PUSH(24, linear) // => `resampler_sinc_push_24_linear`
RESAMPLER_SINC_PUSH(32, linear) // => `resampler_sinc_push_32_linear`
RESAMPLER_SINC_PUSH( 8, cubic) // => `resampler_sinc_push_8_cubic`
RESAMPLER_SINC_PUSH(16, cubic ) // => `resampler_sinc_push_16_cubic`
RESAMPLER_SINC_PUSH(24, cubic) // => `resampler_sinc_push_24_cubic`
RESAMPLER_SINC_PUSH(32, cubic ) // => `resampler_sinc_push_32_cubic`

#define RESAMPLER_SINC_PULL(SIZE, INTERPOLATION)                               \
static int resampler_sinc_pull_##SIZE##_##INTERPOLATION                        \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    Resampler_Sinc * const resampler = state;                                  \
                                                                               \
    int source_frame = 0, target_frame = 0;                                    \
                                                                               \
    while (target_frame < target_frame_count)                                  \
    {                                                                          \
        while (resampler->phase >= resampler->source_delta)                    \
        {                                                                      \
            resampler_sinc_update_windows_##SIZE                               \
            (                                                                  \
                resampler,                                                     \
                source_frame,                                                  \
                source_samples                                                 \
            );                                                                 \
                                                                               \
            resampler->phase -= resampler->source_delta;                       \
                                                                               \
            source_frame++;                                                    \
        }                                                                      \
                                                                               \
        double const blend = resampler->phase * resampler->scale;              \
                                                                               \
        ASSERT(0.0f <= blend && blend < 1.0f);                                 \
                                                                               \
        resample_sinc_frame_##SIZE##_##INTERPOLATION                           \
        (                                                                      \
            resampler,                                                         \
            target_frame,                                                      \
            target_samples,                                                    \
            blend                                                              \
        );                                                                     \
                                                                               \
        resampler->phase += resampler->target_delta;                           \
                                                                               \
        target_frame++;                                                        \
    }                                                                          \
                                                                               \
    ASSERT(resampler->phase >= 0);                                             \
                                                                               \
    return source_frame;                                                       \
}                                                                              \

RESAMPLER_SINC_PULL( 8, linear) // => `resampler_sinc_pull_8_linear`
RESAMPLER_SINC_PULL(16, linear) // => `resampler_sinc_pull_16_linear`
RESAMPLER_SINC_PULL(24, linear) // => `resampler_sinc_pull_24_linear`
RESAMPLER_SINC_PULL(32, linear) // => `resampler_sinc_pull_32_linear`
RESAMPLER_SINC_PULL( 8, cubic ) // => `resampler_sinc_pull_8_cubic`
RESAMPLER_SINC_PULL(16, cubic ) // => `resampler_sinc_pull_16_cubic`
RESAMPLER_SINC_PULL(24, cubic ) // => `resampler_sinc_pull_24_cubic`
RESAMPLER_SINC_PULL(32, cubic ) // => `resampler_sinc_pull_32_cubic`

static Processor resampler_sinc_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration;

    int const * const mode = configuration;

    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);

    int const channel_count = source_format->channel_count;
    long const windows_size = channel_count * sizeof(Window);

    Resampler_Sinc * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_Sinc) + windows_size
    );

    if (resampler)
    {
        static double const latencies [] =
        {
            [resampler_sinc_mode_8_linear ] =  8.0 / 2.0 + 1.0,
            [resampler_sinc_mode_16_linear] = 16.0 / 2.0 + 1.0,
            [resampler_sinc_mode_24_linear] = 24.0 / 2.0 + 1.0,
            [resampler_sinc_mode_32_linear] = 32.0 / 2.0 + 1.0,
            [resampler_sinc_mode_8_cubic  ] =  8.0 / 2.0 + 1.0,
            [resampler_sinc_mode_16_cubic ] = 16.0 / 2.0 + 1.0,
            [resampler_sinc_mode_24_cubic ] = 24.0 / 2.0 + 1.0,
            [resampler_sinc_mode_32_cubic ] = 32.0 / 2.0 + 1.0
        };

        resampler->source_delta = 0x1.0p48 / source_format->sampling_rate;
        resampler->target_delta = 0x1.0p48 / target_format->sampling_rate;
        resampler->phase = resampler->source_delta;
        resampler->latency = latencies[*mode];
        resampler->scale = 1.0 / resampler->source_delta;
        resampler->channel_count = channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < channel_count; channel++)
    {
        static long const window_sizes [] =
        {
            [resampler_sinc_mode_16_linear] = window_16_size,
            [resampler_sinc_mode_16_cubic ] = window_16_size,
            [resampler_sinc_mode_32_linear] = window_32_size,
            [resampler_sinc_mode_32_cubic ] = window_32_size
        };

        float * const samples = alloc(allocator, window_sizes[*mode]);

        if (resampler)
        {
            resampler->windows[channel].samples = samples;
        }

        if (samples)
        {
            memset(samples, 0, window_sizes[*mode]);
        }
    }

    int (* callbacks [][2])(void *, float * restrict, float * restrict, int) =
    {
        { resampler_sinc_push_8_linear , resampler_sinc_pull_8_linear  },
        { resampler_sinc_push_16_linear, resampler_sinc_pull_16_linear },
        { resampler_sinc_push_24_linear, resampler_sinc_pull_24_linear },
        { resampler_sinc_push_32_linear, resampler_sinc_pull_32_linear },
        { resampler_sinc_push_8_cubic  , resampler_sinc_pull_8_cubic   },
        { resampler_sinc_push_16_cubic , resampler_sinc_pull_16_cubic  },
        { resampler_sinc_push_24_cubic , resampler_sinc_pull_24_cubic  },
        { resampler_sinc_push_32_cubic , resampler_sinc_pull_32_cubic  }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_sinc_target_frame_count,
        .pulled_source_frame_count = resampler_sinc_source_frame_count,
        .push = callbacks[*mode][0],
        .pull = callbacks[*mode][1],
        .state = resampler
    };
}

static void resampler_sinc_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (target_format->block_size == 0)
    {
        target_format->block_size = 1;
    }

    ASSERT(target_format->block_size == 1);

    if (source_format->sampling_rate > 0 && target_format->sampling_rate > 0)
    {
        ASSERT(source_format->sampling_rate / target_format->sampling_rate < 2);
        ASSERT(target_format->sampling_rate / source_format->sampling_rate < 2);

        bool const source_format_resolved = source_format->max_block_count > 0;
        bool const target_format_resolved = target_format->max_block_count > 0;

        if (source_format->block_size > 0 && target_format->block_size > 0)
        {
            if (!source_format_resolved && target_format_resolved)
            {
                long count = target_format->max_block_count;

                count *= target_format->block_size;
                count *= source_format->sampling_rate;
                count /= target_format->sampling_rate;
                count /= source_format->block_size;

                source_format->max_block_count = (int)count;
            }
            else if (source_format_resolved && !target_format_resolved)
            {
                long count = source_format->max_block_count;

                count *= source_format->block_size;
                count *= target_format->sampling_rate;
                count += source_format->sampling_rate - 1;
                count /= source_format->sampling_rate;
                count /= target_format->block_size;

                target_format->max_block_count = (int)count;
            }
        }

        ASSERT
        (
             source_format->block_size
                * source_format->max_block_count
                    * target_format->sampling_rate
          <
             target_format->block_size
                * target_format->max_block_count
                    * source_format->sampling_rate
        );

        ASSERT
        (
             source_format->block_size
                * (source_format->max_block_count + 1)
                    * target_format->sampling_rate
          >=
             target_format->block_size
                * target_format->max_block_count
                    * source_format->sampling_rate
        );
    }

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
}

#define RESAMPLER_SINC(SIZE, INTERPOLATION)                                    \
static const Pass resampler_sinc_##SIZE##_##INTERPOLATION =                    \
{                                                                              \
    .create_processor = resampler_sinc_create_processor,                       \
    .restrain_formats = resampler_sinc_restrain_formats,                       \
    .configuration = &resampler_sinc_mode_##SIZE##_##INTERPOLATION             \
};                                                                             \

RESAMPLER_SINC( 8, linear) // => `resampler_sinc_8_linear`
RESAMPLER_SINC(16, linear) // => `resampler_sinc_16_linear`
RESAMPLER_SINC(24, linear) // => `resampler_sinc_24_linear`
RESAMPLER_SINC(32, linear) // => `resampler_sinc_32_linear`
RESAMPLER_SINC( 8, cubic ) // => `resampler_sinc_8_cubic`
RESAMPLER_SINC(16, cubic ) // => `resampler_sinc_16_cubic`
RESAMPLER_SINC(24, cubic ) // => `resampler_sinc_24_cubic`
RESAMPLER_SINC(32, cubic ) // => `resampler_sinc_32_cubic`

/* -------------------------------------------------------------------------- */

#pragma mark - 2X FIR Resampling

static float const kernel_2x_fir_16 [8];
static float const kernel_2x_fir_32 [16];
static float const kernel_2x_fir_48 [24];
static float const kernel_2x_fir_64 [32];
static float const kernel_2x_fir_80 [40];
static float const kernel_2x_fir_96 [48];

static int const resampler_2x_fir_order_16 = 0;
static int const resampler_2x_fir_order_32 = 1;
static int const resampler_2x_fir_order_48 = 2;
static int const resampler_2x_fir_order_64 = 3;
static int const resampler_2x_fir_order_80 = 4;
static int const resampler_2x_fir_order_96 = 5;

typedef struct Resampler_2X_FIR
{
    double latency;
    int channel_count;
    int index;
    Window windows [];
}
Resampler_2X_FIR;

static int resampler_2x_fir_frame_count_mul_2
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Resampler_2X_FIR const * const resampler = state;

    *latency += resampler->latency;

    return frame_count * 2;
}

static int resampler_2x_fir_frame_count_div_2
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Resampler_2X_FIR const * const resampler = state;

    *latency += resampler->latency;

    return frame_count / 2;
}

#define UPSAMPLE_2X_FIR(ORDER, HALF, QUARTER)                                  \
static void upsample_2x_fir_##ORDER                                            \
    (                                                                          \
        Resampler_2X_FIR * const resampler,                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const channel_count = resampler->channel_count;                        \
                                                                               \
    for (int frame = 0; frame < source_frame_count; frame++)                   \
    {                                                                          \
        for (int channel = 0; channel < channel_count; channel++)              \
        {                                                                      \
            int const index = frame * channel_count + channel;                 \
                                                                               \
            float const input_sample = source_samples[index];                  \
                                                                               \
            Window * const window = &resampler->windows[channel];              \
                                                                               \
            window_##HALF##_write(window, input_sample, resampler->index);     \
                                                                               \
            float const * const slice = window->samples + resampler->index;    \
                                                                               \
            float const * const kernel = kernel_2x_fir_##ORDER;                \
                                                                               \
            float const acc_0 = slice[QUARTER] * 0.5f;                         \
            float const acc_1 = convolve_##HALF(kernel, slice + 1);            \
                                                                               \
            float const output_sample_0 = acc_0 * 2.0f;                        \
            float const output_sample_1 = acc_1 * 2.0f;                        \
                                                                               \
            int const index_0 = (frame * 2 + 0) * channel_count + channel;     \
            int const index_1 = (frame * 2 + 1) * channel_count + channel;     \
                                                                               \
            target_samples[index_0] = output_sample_0;                         \
            target_samples[index_1] = output_sample_1;                         \
        }                                                                      \
                                                                               \
        resampler->index = window_##HALF##_next_index(resampler->index);       \
    }                                                                          \
}                                                                              \

UPSAMPLE_2X_FIR(16,  8,  4) // => `upsample_2x_fir_16`
UPSAMPLE_2X_FIR(32, 16,  8) // => `upsample_2x_fir_32`
UPSAMPLE_2X_FIR(48, 24, 12) // => `upsample_2x_fir_48`
UPSAMPLE_2X_FIR(64, 32, 16) // => `upsample_2x_fir_64`
UPSAMPLE_2X_FIR(80, 40, 20) // => `upsample_2x_fir_80`
UPSAMPLE_2X_FIR(96, 48, 24) // => `upsample_2x_fir_96`

#define DOWNSAMPLE_2X_FIR(ORDER, HALF, QUARTER)                                \
static void downsample_2x_fir_##ORDER                                          \
    (                                                                          \
        Resampler_2X_FIR * const resampler,                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const channel_count = resampler->channel_count;                        \
                                                                               \
    for (int frame = 0; frame < target_frame_count; frame++)                   \
    {                                                                          \
        for (int channel = 0; channel < channel_count; channel++)              \
        {                                                                      \
            int const index_0 = (frame * 2 + 0) * channel_count + channel;     \
            int const index_1 = (frame * 2 + 1) * channel_count + channel;     \
                                                                               \
            float const input_sample_0 = source_samples[index_0];              \
            float const input_sample_1 = source_samples[index_1];              \
                                                                               \
            Window * const window_0 = &resampler->windows[channel * 2 + 0];    \
            Window * const window_1 = &resampler->windows[channel * 2 + 1];    \
                                                                               \
            float const * const slice_0 = window_0->samples + resampler->index;\
            float const * const slice_1 = window_1->samples + resampler->index;\
                                                                               \
            float const * const kernel = kernel_2x_fir_##ORDER;                \
                                                                               \
            float const acc_0 = slice_0[QUARTER] * 0.5f;                       \
            float const acc_1 = convolve_##HALF(kernel, slice_1);              \
                                                                               \
            window_##HALF##_write(window_0, input_sample_0, resampler->index); \
            window_##HALF##_write(window_1, input_sample_1, resampler->index); \
                                                                               \
            float const output_sample = acc_0 + acc_1;                         \
                                                                               \
            int const index = frame * channel_count + channel;                 \
                                                                               \
            target_samples[index] = output_sample;                             \
        }                                                                      \
                                                                               \
        resampler->index = window_##HALF##_next_index(resampler->index);       \
    }                                                                          \
}                                                                              \

DOWNSAMPLE_2X_FIR(16,  8,  4) // => `downsample_2x_fir_16`
DOWNSAMPLE_2X_FIR(32, 16,  8) // => `downsample_2x_fir_32`
DOWNSAMPLE_2X_FIR(48, 24, 12) // => `downsample_2x_fir_48`
DOWNSAMPLE_2X_FIR(64, 32, 16) // => `downsample_2x_fir_64`
DOWNSAMPLE_2X_FIR(80, 40, 20) // => `downsample_2x_fir_80`
DOWNSAMPLE_2X_FIR(96, 48, 24) // => `downsample_2x_fir_96`

#define UPSAMPLER_2X_FIR_PUSH(ORDER)                                           \
static int upsampler_2x_fir_push_##ORDER                                       \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const target_frame_count = source_frame_count * 2;                     \
                                                                               \
    upsample_2x_fir_##ORDER                                                    \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        source_frame_count                                                     \
    );                                                                         \
                                                                               \
    return target_frame_count;                                                 \
}                                                                              \

UPSAMPLER_2X_FIR_PUSH(16) // => `upsampler_2x_fir_push_16`
UPSAMPLER_2X_FIR_PUSH(32) // => `upsampler_2x_fir_push_32`
UPSAMPLER_2X_FIR_PUSH(48) // => `upsampler_2x_fir_push_48`
UPSAMPLER_2X_FIR_PUSH(64) // => `upsampler_2x_fir_push_64`
UPSAMPLER_2X_FIR_PUSH(80) // => `upsampler_2x_fir_push_80`
UPSAMPLER_2X_FIR_PUSH(96) // => `upsampler_2x_fir_push_96`

#define UPSAMPLER_2X_FIR_PULL(ORDER)                                           \
static int upsampler_2x_fir_pull_##ORDER                                       \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const source_frame_count = target_frame_count / 2;                     \
                                                                               \
    upsample_2x_fir_##ORDER                                                    \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        source_frame_count                                                     \
    );                                                                         \
                                                                               \
    return source_frame_count;                                                 \
}                                                                              \

UPSAMPLER_2X_FIR_PULL(16) // => `upsampler_2x_fir_pull_16`
UPSAMPLER_2X_FIR_PULL(32) // => `upsampler_2x_fir_pull_32`
UPSAMPLER_2X_FIR_PULL(48) // => `upsampler_2x_fir_pull_48`
UPSAMPLER_2X_FIR_PULL(64) // => `upsampler_2x_fir_pull_64`
UPSAMPLER_2X_FIR_PULL(80) // => `upsampler_2x_fir_pull_80`
UPSAMPLER_2X_FIR_PULL(96) // => `upsampler_2x_fir_pull_96`

#define DOWNSAMPLER_2X_FIR_PUSH(ORDER)                                         \
static int downsampler_2x_fir_push_##ORDER                                     \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const source_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const target_frame_count = source_frame_count / 2;                     \
                                                                               \
    downsample_2x_fir_##ORDER                                                  \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        target_frame_count                                                     \
    );                                                                         \
                                                                               \
    return target_frame_count;                                                 \
}                                                                              \

DOWNSAMPLER_2X_FIR_PUSH(16) // => `downsampler_2x_fir_push_16`
DOWNSAMPLER_2X_FIR_PUSH(32) // => `downsampler_2x_fir_push_32`
DOWNSAMPLER_2X_FIR_PUSH(48) // => `downsampler_2x_fir_push_48`
DOWNSAMPLER_2X_FIR_PUSH(64) // => `downsampler_2x_fir_push_64`
DOWNSAMPLER_2X_FIR_PUSH(80) // => `downsampler_2x_fir_push_80`
DOWNSAMPLER_2X_FIR_PUSH(96) // => `downsampler_2x_fir_push_96`

#define DOWNSAMPLER_2X_FIR_PULL(ORDER)                                         \
static int downsampler_2x_fir_pull_##ORDER                                     \
    (                                                                          \
        void * const state,                                                    \
        float * restrict const source_samples,                                 \
        float * restrict const target_samples,                                 \
        int const target_frame_count                                           \
    )                                                                          \
{                                                                              \
    int const source_frame_count = target_frame_count * 2;                     \
                                                                               \
    downsample_2x_fir_##ORDER                                                  \
    (                                                                          \
        state,                                                                 \
        source_samples,                                                        \
        target_samples,                                                        \
        target_frame_count                                                     \
    );                                                                         \
                                                                               \
    return source_frame_count;                                                 \
}                                                                              \

DOWNSAMPLER_2X_FIR_PULL(16) // => `downsampler_2x_fir_pull_16`
DOWNSAMPLER_2X_FIR_PULL(32) // => `downsampler_2x_fir_pull_32`
DOWNSAMPLER_2X_FIR_PULL(48) // => `downsampler_2x_fir_pull_48`
DOWNSAMPLER_2X_FIR_PULL(64) // => `downsampler_2x_fir_pull_64`
DOWNSAMPLER_2X_FIR_PULL(80) // => `downsampler_2x_fir_pull_80`
DOWNSAMPLER_2X_FIR_PULL(96) // => `downsampler_2x_fir_pull_96`

static Processor upsampler_2x_fir_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    int const * const order = configuration;

    ASSERT(source_format->sampling_rate * 2 == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size * 2 == target_format->block_size);
    ASSERT(target_format->max_block_count == target_format->max_block_count);

    long const windows_size = source_format->channel_count * sizeof(Window);

    Resampler_2X_FIR * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_2X_FIR) + windows_size
    );

    if (resampler)
    {
        double const latencies [] =
        {
            [resampler_2x_fir_order_16] =  8.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_32] = 16.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_48] = 24.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_64] = 32.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_80] = 40.0 / target_format->sampling_rate,
            [resampler_2x_fir_order_96] = 48.0 / target_format->sampling_rate
        };

        resampler->latency = latencies[*order];
        resampler->channel_count = source_format->channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < source_format->channel_count; channel++)
    {
        long window_size = 0;

        switch (*order)
        {
          case resampler_2x_fir_order_16: window_size = window_8_size; break;
          case resampler_2x_fir_order_32: window_size = window_16_size; break;
          case resampler_2x_fir_order_48: window_size = window_24_size; break;
          case resampler_2x_fir_order_64: window_size = window_32_size; break;
          case resampler_2x_fir_order_80: window_size = window_40_size; break;
          case resampler_2x_fir_order_96: window_size = window_48_size; break;
        }

        float * const samples = alloc(allocator, window_size);

        if (resampler)
        {
            resampler->windows[channel].samples = samples;
        }

        if (samples)
        {
            memset(samples, 0, window_size);
        }
    }

    int (* callbacks [][2])(void *, float * restrict, float * restrict, int) =
    {
        { upsampler_2x_fir_push_16, upsampler_2x_fir_pull_16 },
        { upsampler_2x_fir_push_32, upsampler_2x_fir_pull_32 },
        { upsampler_2x_fir_push_48, upsampler_2x_fir_pull_48 },
        { upsampler_2x_fir_push_64, upsampler_2x_fir_pull_64 },
        { upsampler_2x_fir_push_80, upsampler_2x_fir_pull_80 },
        { upsampler_2x_fir_push_96, upsampler_2x_fir_pull_96 }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_2x_fir_frame_count_mul_2,
        .pulled_source_frame_count = resampler_2x_fir_frame_count_div_2,
        .push = callbacks[*order][0],
        .pull = callbacks[*order][1],
        .state = resampler
    };
}

static Processor downsampler_2x_fir_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    int const * const order = configuration;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate * 2);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);
    ASSERT(source_format->block_size == target_format->block_size * 2);
    ASSERT(target_format->max_block_count == target_format->max_block_count);

    long const windows_size = source_format->channel_count * 2 * sizeof(Window);

    Resampler_2X_FIR * const resampler = alloc
    (
        allocator,
        sizeof(Resampler_2X_FIR) + windows_size
    );

    if (resampler)
    {
        double const latencies [] =
        {
            [resampler_2x_fir_order_16] =  8.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_32] = 16.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_48] = 24.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_64] = 32.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_80] = 40.0 / source_format->sampling_rate,
            [resampler_2x_fir_order_96] = 48.0 / source_format->sampling_rate
        };

        resampler->latency = latencies[*order];
        resampler->channel_count = source_format->channel_count;
        resampler->index = 0;
    }

    for (int channel = 0; channel < source_format->channel_count; channel++)
    {
        long const window_sizes [] =
        {
            [resampler_2x_fir_order_16] = window_8_size,
            [resampler_2x_fir_order_32] = window_16_size,
            [resampler_2x_fir_order_48] = window_24_size,
            [resampler_2x_fir_order_64] = window_32_size,
            [resampler_2x_fir_order_80] = window_40_size,
            [resampler_2x_fir_order_96] = window_48_size
        };

        float * const samples_0 = alloc(allocator, window_sizes[*order]);
        float * const samples_1 = alloc(allocator, window_sizes[*order]);

        if (resampler)
        {
            resampler->windows[channel * 2 + 0].samples = samples_0;
            resampler->windows[channel * 2 + 1].samples = samples_1;
        }

        if (samples_0 && samples_1)
        {
            memset(samples_0, 0, window_sizes[*order]);
            memset(samples_1, 0, window_sizes[*order]);
        }
    }

    int (* callbacks [][2])(void *, float * restrict, float * restrict, int) =
    {
        { downsampler_2x_fir_push_16, downsampler_2x_fir_pull_16 },
        { downsampler_2x_fir_push_32, downsampler_2x_fir_pull_32 },
        { downsampler_2x_fir_push_48, downsampler_2x_fir_pull_48 },
        { downsampler_2x_fir_push_64, downsampler_2x_fir_pull_64 },
        { downsampler_2x_fir_push_80, downsampler_2x_fir_pull_80 },
        { downsampler_2x_fir_push_96, downsampler_2x_fir_pull_96 }
    };

    return (Processor)
    {
        .pushed_target_frame_count = resampler_2x_fir_frame_count_div_2,
        .pulled_source_frame_count = resampler_2x_fir_frame_count_mul_2,
        .push = callbacks[*order][0],
        .pull = callbacks[*order][1],
        .state = resampler
    };
}

static void upsampler_2x_fir_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->sampling_rate == 0 && target_format->sampling_rate > 0)
    {
        source_format->sampling_rate = target_format->sampling_rate / 2;
    }

    if (source_format->sampling_rate > 0 && target_format->sampling_rate == 0)
    {
        target_format->sampling_rate = source_format->sampling_rate * 2;
    }

    ASSERT(source_format->sampling_rate * 2 == target_format->sampling_rate);

    if (source_format->block_size == 0 && target_format->block_size > 0)
    {
        source_format->block_size = target_format->block_size / 2;
    }

    if (source_format->block_size > 0 && target_format->block_size == 0)
    {
        target_format->block_size = source_format->block_size * 2;
    }

    ASSERT(source_format->block_size * 2 == target_format->block_size);

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

static void downsampler_2x_fir_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->sampling_rate == 0 && target_format->sampling_rate > 0)
    {
        source_format->sampling_rate = target_format->sampling_rate * 2;
    }

    if (source_format->sampling_rate > 0 && target_format->sampling_rate == 0)
    {
        target_format->sampling_rate = source_format->sampling_rate / 2;
    }

    ASSERT(source_format->sampling_rate == target_format->sampling_rate * 2);

    if (source_format->block_size == 0 && target_format->block_size > 0)
    {
        source_format->block_size = target_format->block_size * 2;
    }

    if (source_format->block_size > 0 && target_format->block_size == 0)
    {
        target_format->block_size = source_format->block_size / 2;
    }

    ASSERT(source_format->block_size == target_format->block_size * 2);

    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
    unify(&source_format->max_block_count, &target_format->max_block_count);
}

#define UPSAMPLER_2X_FIR(ORDER)                                                \
static const Pass upsampler_2x_fir_##ORDER =                                   \
{                                                                              \
    .create_processor = upsampler_2x_fir_create_processor,                     \
    .restrain_formats = upsampler_2x_fir_restrain_formats,                     \
    .configuration = &resampler_2x_fir_order_##ORDER                           \
};                                                                             \

UPSAMPLER_2X_FIR(16) // <= `upsampler_2x_fir_16`
UPSAMPLER_2X_FIR(32) // <= `upsampler_2x_fir_32`
UPSAMPLER_2X_FIR(48) // <= `upsampler_2x_fir_48`
UPSAMPLER_2X_FIR(64) // <= `upsampler_2x_fir_64`
UPSAMPLER_2X_FIR(80) // <= `upsampler_2x_fir_80`
UPSAMPLER_2X_FIR(96) // <= `upsampler_2x_fir_96`

#define DOWNSAMPLER_2X_FIR(ORDER)                                              \
static const Pass downsampler_2x_fir_##ORDER =                                 \
{                                                                              \
    .create_processor = downsampler_2x_fir_create_processor,                   \
    .restrain_formats = downsampler_2x_fir_restrain_formats,                   \
.configuration = &resampler_2x_fir_order_##ORDER                               \
};                                                                             \

DOWNSAMPLER_2X_FIR(16) // <= `downsampler_2x_fir_16`
DOWNSAMPLER_2X_FIR(32) // <= `downsampler_2x_fir_32`
DOWNSAMPLER_2X_FIR(48) // <= `downsampler_2x_fir_48`
DOWNSAMPLER_2X_FIR(64) // <= `downsampler_2x_fir_64`
DOWNSAMPLER_2X_FIR(80) // <= `downsampler_2x_fir_80`
DOWNSAMPLER_2X_FIR(96) // <= `downsampler_2x_fir_96`

/* -------------------------------------------------------------------------- */

#pragma mark - Slicing

typedef struct Slicer
{
    double latency;
    int channel_count;
    int block_size;
    int index;
    _Alignas(cache_line_size) float window [];
}
Slicer;

static int slicer_frame_count // <- Same for the push/pull versions.
    (
        void const * const state,
        int const frame_count,
        double * const latency
    )
{
    Slicer const * const slicer = state;

    int const block_count = (slicer->index + frame_count) / slicer->block_size;

    *latency += slicer->latency; // TODO: Fix this!

    return block_count * slicer->block_size;
}

static int slicer_push
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const source_frame_count
    )
{
    Slicer * const slicer = state;

    int const block_size = slicer->block_size;

    int source_frame = 0, target_frame_count = 0;

    if (source_frame < source_frame_count)
    {
        int const frames_left_in_block = block_size - slicer->index;
        int const frame_count = min(frames_left_in_block, source_frame_count);

        memcpy
        (
            slicer->window + slicer->index * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        source_frame += frame_count;
        slicer->index += frame_count;
    }

    if (slicer->index == block_size)
    {
        int const frames_left = source_frame_count - source_frame;
        int const block_count = frames_left / block_size;

        target_frame_count = (block_count + 1) * block_size;

        memcpy
        (
            target_samples,
            slicer->window,
            block_size * slicer->channel_count * sizeof(float)
        );

        memcpy
        (
            target_samples + block_size * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            block_count * block_size * slicer->channel_count * sizeof(float)
        );

        source_frame += block_count * block_size;
        slicer->index = 0;
    }

    if (source_frame < source_frame_count)
    {
        int const frames_left = source_frame_count - source_frame;
        int const frame_count = frames_left;

        memcpy
        (
            slicer->window + slicer->index * slicer->channel_count,
            source_samples + source_frame * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        source_frame += frame_count;
        slicer->index += frame_count;
    }

    ASSERT(source_frame == source_frame_count);

    return target_frame_count;
}

static int slicer_pull
    (
        void * const state,
        float * restrict const source_samples,
        float * restrict const target_samples,
        int const target_frame_count
    )
{
    Slicer * const slicer = state;

    int const block_size = slicer->block_size;

    int target_frame = 0, source_frame_count = 0;

    if (target_frame < target_frame_count)
    {
        int const frames_left_in_block = block_size - slicer->index;
        int const frame_count = min(frames_left_in_block, target_frame_count);

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            slicer->window + slicer->index * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        target_frame += frame_count;
        slicer->index += frame_count;
    }

    if (slicer->index == block_size)
    {
        int const frames_left = target_frame_count - target_frame;
        int const block_count = frames_left / block_size;

        source_frame_count = (block_count + 1) * block_size;

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            source_samples,
            block_count * block_size * slicer->channel_count * sizeof(float)
        );

        memcpy
        (
            slicer->window,
            source_samples + block_count * block_size * slicer->channel_count,
            block_size * slicer->channel_count * sizeof(float)
        );

        target_frame += block_count * block_size;
        slicer->index = 0;
    }

    if (target_frame < target_frame_count)
    {
        int const frames_left = target_frame_count - target_frame;
        int const frame_count = frames_left;

        memcpy
        (
            target_samples + target_frame * slicer->channel_count,
            slicer->window + slicer->index * slicer->channel_count,
            frame_count * slicer->channel_count * sizeof(float)
        );

        target_frame += frame_count;
        slicer->index += frame_count;
    }

    ASSERT(target_frame == target_frame_count);

    return source_frame_count;
}

static Processor slicer_create_processor
    (
        void const * const configuration,
        Bump_Allocator * const allocator,
        Format const * const source_format,
        Format const * const target_format
    )
{
    (void)configuration;

    ASSERT(source_format->sampling_rate == target_format->sampling_rate);
    ASSERT(source_format->channel_count == target_format->channel_count);
    ASSERT(source_format->layout == layout_interleaved);
    ASSERT(target_format->layout == layout_interleaved);

    int const source_block_size = source_format->block_size;
    int const target_block_size = target_format->block_size;

    int const channel_count = source_format->channel_count;
    int const max_block_size = max(source_block_size, target_block_size);
    int const window_size = max_block_size * channel_count * sizeof(float);

    Slicer * const slicer = alloc
    (
        allocator,
        sizeof(Slicer) + window_size
    );

    if (slicer)
    {
        slicer->latency = (max_block_size / 2.0) / source_format->sampling_rate;
        slicer->channel_count = channel_count;
        slicer->block_size = max_block_size;
        slicer->index = max_block_size / 2;
        memset(slicer->window, 0, window_size);
    }

    return (Processor)
    {
        .pushed_target_frame_count = slicer_frame_count,
        .pulled_source_frame_count = slicer_frame_count,
        .push = slicer_push,
        .pull = slicer_pull,
        .state = slicer
    };
}

static void slicer_restrain_formats
    (
        void const * const configuration,
        Format * restrict const source_format,
        Format * restrict const target_format
    )
{
    (void)configuration;

    if (source_format->block_size > 0 && target_format->block_size > 0)
    {
        bool const source_format_resolved = source_format->max_block_count > 0;
        bool const target_format_resolved = target_format->max_block_count > 0;

        if (!source_format_resolved && target_format_resolved)
        {
            long count = target_format->max_block_count;

            count *= target_format->block_size;
            count /= source_format->block_size;

            source_format->max_block_count = (int)count;
        }
        else if (source_format_resolved && !target_format_resolved)
        {
            long count = source_format->max_block_count;

            count *= source_format->block_size;
            count += target_format->block_size - 1;
            count /= target_format->block_size;

            target_format->max_block_count = (int)count;
        }

        ASSERT
        (
             source_format->block_size * source_format->max_block_count
          <=
             target_format->block_size * target_format->max_block_count
        );

        ASSERT
        (
             source_format->block_size * (source_format->max_block_count + 1)
          >=
             target_format->block_size * target_format->max_block_count
        );
    }

    unify(&source_format->sampling_rate, &target_format->sampling_rate);
    unify(&source_format->channel_count, &target_format->channel_count);
    unify(&source_format->layout, &target_format->layout);
}

static const Pass slicer =
{
    .create_processor = slicer_create_processor,
    .restrain_formats = slicer_restrain_formats
};

/* -------------------------------------------------------------------------- */

#pragma mark - Passes

enum
{
    pass_tag_interleaver,
    pass_tag_deinterleaver,
    pass_tag_upsampler_2x_fir_16,
    pass_tag_upsampler_2x_fir_32,
    pass_tag_upsampler_2x_fir_48,
    pass_tag_upsampler_2x_fir_64,
    pass_tag_upsampler_2x_fir_80,
    pass_tag_upsampler_2x_fir_96,
    pass_tag_downsampler_2x_fir_16,
    pass_tag_downsampler_2x_fir_32,
    pass_tag_downsampler_2x_fir_48,
    pass_tag_downsampler_2x_fir_64,
    pass_tag_downsampler_2x_fir_80,
    pass_tag_downsampler_2x_fir_96,
    pass_tag_resampler_sinc_8_linear,
    pass_tag_resampler_sinc_16_linear,
    pass_tag_resampler_sinc_24_linear,
    pass_tag_resampler_sinc_32_linear,
    pass_tag_resampler_sinc_8_cubic,
    pass_tag_resampler_sinc_16_cubic,
    pass_tag_resampler_sinc_24_cubic,
    pass_tag_resampler_sinc_32_cubic,
    pass_tag_slicer
};

static Pass make_pass (short pass_tag)
{
    switch (pass_tag)
    {
      case pass_tag_interleaver: return interleaver;
      case pass_tag_deinterleaver: return deinterleaver;
      case pass_tag_upsampler_2x_fir_16: return upsampler_2x_fir_16;
      case pass_tag_upsampler_2x_fir_32: return upsampler_2x_fir_32;
      case pass_tag_upsampler_2x_fir_48: return upsampler_2x_fir_48;
      case pass_tag_upsampler_2x_fir_64: return upsampler_2x_fir_64;
      case pass_tag_upsampler_2x_fir_80: return upsampler_2x_fir_80;
      case pass_tag_upsampler_2x_fir_96: return upsampler_2x_fir_96;
      case pass_tag_downsampler_2x_fir_16: return downsampler_2x_fir_16;
      case pass_tag_downsampler_2x_fir_32: return downsampler_2x_fir_32;
      case pass_tag_downsampler_2x_fir_48: return downsampler_2x_fir_48;
      case pass_tag_downsampler_2x_fir_64: return downsampler_2x_fir_64;
      case pass_tag_downsampler_2x_fir_80: return downsampler_2x_fir_80;
      case pass_tag_downsampler_2x_fir_96: return downsampler_2x_fir_96;
      case pass_tag_resampler_sinc_8_linear: return resampler_sinc_8_linear;
      case pass_tag_resampler_sinc_16_linear: return resampler_sinc_16_linear;
      case pass_tag_resampler_sinc_24_linear: return resampler_sinc_24_linear;
      case pass_tag_resampler_sinc_32_linear: return resampler_sinc_32_linear;
      case pass_tag_resampler_sinc_8_cubic: return resampler_sinc_8_cubic;
      case pass_tag_resampler_sinc_16_cubic: return resampler_sinc_16_cubic;
      case pass_tag_resampler_sinc_24_cubic: return resampler_sinc_24_cubic;
      case pass_tag_resampler_sinc_32_cubic: return resampler_sinc_32_cubic;
      case pass_tag_slicer: return slicer;
    }

    ASSERT(false);

    return (Pass) { 0 };
}

static void invert_passes (short passes [const], int const pass_count)
{
    short const inversions [] =
    {
        [pass_tag_interleaver] = pass_tag_deinterleaver,
        [pass_tag_deinterleaver] = pass_tag_interleaver,
        [pass_tag_upsampler_2x_fir_16] = pass_tag_downsampler_2x_fir_16,
        [pass_tag_upsampler_2x_fir_32] = pass_tag_downsampler_2x_fir_32,
        [pass_tag_upsampler_2x_fir_48] = pass_tag_downsampler_2x_fir_48,
        [pass_tag_upsampler_2x_fir_64] = pass_tag_downsampler_2x_fir_64,
        [pass_tag_upsampler_2x_fir_80] = pass_tag_downsampler_2x_fir_80,
        [pass_tag_upsampler_2x_fir_96] = pass_tag_downsampler_2x_fir_96,
        [pass_tag_downsampler_2x_fir_16] = pass_tag_upsampler_2x_fir_16,
        [pass_tag_downsampler_2x_fir_32] = pass_tag_upsampler_2x_fir_32,
        [pass_tag_downsampler_2x_fir_48] = pass_tag_upsampler_2x_fir_48,
        [pass_tag_downsampler_2x_fir_64] = pass_tag_upsampler_2x_fir_64,
        [pass_tag_downsampler_2x_fir_80] = pass_tag_upsampler_2x_fir_80,
        [pass_tag_downsampler_2x_fir_96] = pass_tag_upsampler_2x_fir_96,
        [pass_tag_resampler_sinc_8_linear] = pass_tag_resampler_sinc_8_linear,
        [pass_tag_resampler_sinc_16_linear] = pass_tag_resampler_sinc_16_linear,
        [pass_tag_resampler_sinc_24_linear] = pass_tag_resampler_sinc_24_linear,
        [pass_tag_resampler_sinc_32_linear] = pass_tag_resampler_sinc_32_linear,
        [pass_tag_resampler_sinc_8_cubic] = pass_tag_resampler_sinc_8_cubic,
        [pass_tag_resampler_sinc_16_cubic] = pass_tag_resampler_sinc_16_cubic,
        [pass_tag_resampler_sinc_24_cubic] = pass_tag_resampler_sinc_24_cubic,
        [pass_tag_resampler_sinc_32_cubic] = pass_tag_resampler_sinc_32_cubic,
        [pass_tag_slicer] = pass_tag_slicer
    };

    for (int index = 0; index < pass_count; index++)
    {
        passes[index] = inversions[passes[index]];
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Conversion Resolution

enum { max_pass_count = 10, max_format_count = max_pass_count + 1 };

typedef struct Sampling_Rate_Conversion
{
    int up_2x_pass_count;
    int down_2x_pass_count;
    bool needs_fractional_resampling;
}
Sampling_Rate_Conversion;

static void resolve_sampling_rate_conversion
    (
        int const source_sampling_rate,
        int const target_sampling_rate,
        bool const oversample,
        Sampling_Rate_Conversion * const sampling_rate_conversion
    )
{
    int factor = 0;

    if (source_sampling_rate < target_sampling_rate)
    {
        while
        (
            (float)target_sampling_rate / (source_sampling_rate * (1 << factor))
          >
            (float)(source_sampling_rate * (2 << factor)) / target_sampling_rate
        )
        {
            factor += 1;
        }

        sampling_rate_conversion->up_2x_pass_count = factor;

        if (source_sampling_rate * (1 << factor) != target_sampling_rate)
        {
            sampling_rate_conversion->needs_fractional_resampling = true;

            if (oversample)
            {
                sampling_rate_conversion->up_2x_pass_count += 1;
                sampling_rate_conversion->down_2x_pass_count += 1;
            }
        }
    }
    else // target_sampling_rate > source_sampling_rate
    {
        while
        (
            (float)source_sampling_rate / (target_sampling_rate * (1 << factor))
          >
            (float)(target_sampling_rate * (2 << factor)) / source_sampling_rate
        )
        {
            factor += 1;
        }

        sampling_rate_conversion->down_2x_pass_count = factor;

        if (target_sampling_rate * (1 << factor) != source_sampling_rate)
        {
            sampling_rate_conversion->needs_fractional_resampling = true;

            if (oversample)
            {
                sampling_rate_conversion->up_2x_pass_count += 1;
                sampling_rate_conversion->down_2x_pass_count += 1;
            }
        }
    }
}

typedef struct DrB_Audio_Conversion
{
    Format formats [max_format_count];
    short passes [max_pass_count];
    int pass_count, format_count;
}
DrB_Audio_Conversion;

static int resolve_passes
    (
        Format const * const source_format,
        Format const * const target_format,
        DrB_Audio_Converter_Quality const quality,
        short passes [const]
    )
{
    enum { max_2x_pass_count = 6 };

    static short const upsamplers_2x [][max_2x_pass_count] =
    {
        [drb_audio_converter_quality_poor] =
        {
            pass_tag_upsampler_2x_fir_48, pass_tag_upsampler_2x_fir_32,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_fine] =
        {
            pass_tag_upsampler_2x_fir_64, pass_tag_upsampler_2x_fir_48,
            pass_tag_upsampler_2x_fir_32, pass_tag_upsampler_2x_fir_16,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_good] =
        {
            pass_tag_upsampler_2x_fir_80, pass_tag_upsampler_2x_fir_64,
            pass_tag_upsampler_2x_fir_48, pass_tag_upsampler_2x_fir_32,
            pass_tag_upsampler_2x_fir_16, pass_tag_upsampler_2x_fir_16
        },
        [drb_audio_converter_quality_best] =
        {
            pass_tag_upsampler_2x_fir_96, pass_tag_upsampler_2x_fir_80,
            pass_tag_upsampler_2x_fir_64, pass_tag_upsampler_2x_fir_48,
            pass_tag_upsampler_2x_fir_32, pass_tag_upsampler_2x_fir_16
        }
    };

    static short const downsamplers_2x [][max_2x_pass_count] =
    {
        [drb_audio_converter_quality_poor] =
        {
            pass_tag_downsampler_2x_fir_48, pass_tag_downsampler_2x_fir_32,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_fine] =
        {
            pass_tag_downsampler_2x_fir_64, pass_tag_downsampler_2x_fir_48,
            pass_tag_downsampler_2x_fir_32, pass_tag_downsampler_2x_fir_16,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_good] =
        {
            pass_tag_downsampler_2x_fir_80, pass_tag_downsampler_2x_fir_64,
            pass_tag_downsampler_2x_fir_48, pass_tag_downsampler_2x_fir_32,
            pass_tag_downsampler_2x_fir_16, pass_tag_downsampler_2x_fir_16
        },
        [drb_audio_converter_quality_best] =
        {
            pass_tag_downsampler_2x_fir_96, pass_tag_downsampler_2x_fir_80,
            pass_tag_downsampler_2x_fir_64, pass_tag_downsampler_2x_fir_48,
            pass_tag_downsampler_2x_fir_32, pass_tag_downsampler_2x_fir_16
        }
    };

    static short const resamplers [] =
    {
        [drb_audio_converter_quality_poor] = pass_tag_resampler_sinc_16_linear,
        [drb_audio_converter_quality_fine] = pass_tag_resampler_sinc_16_cubic,
        [drb_audio_converter_quality_good] = pass_tag_resampler_sinc_32_linear,
        [drb_audio_converter_quality_best] = pass_tag_resampler_sinc_32_cubic
    };

    Sampling_Rate_Conversion sampling_rate_conversion = { 0 };

    bool const needs_oversampling = quality > drb_audio_converter_quality_poor;

    resolve_sampling_rate_conversion
    (
        source_format->sampling_rate,
        target_format->sampling_rate,
        needs_oversampling,
        &sampling_rate_conversion
    );

    ASSERT(sampling_rate_conversion.up_2x_pass_count < max_2x_pass_count);
    ASSERT(sampling_rate_conversion.down_2x_pass_count < max_2x_pass_count);

    int pass_count = 0;
    int up_2x_index = 0;
    int down_2x_index = sampling_rate_conversion.down_2x_pass_count - 1;

    if (source_format->layout != layout_interleaved)
    {
        if (source_format->channel_count > 1)
        {
            passes[pass_count++] = pass_tag_interleaver;
        }
    }

    for (int it = 0; it < sampling_rate_conversion.up_2x_pass_count; it++)
    {
        passes[pass_count++] = upsamplers_2x[quality][up_2x_index++];
    }

    if (sampling_rate_conversion.needs_fractional_resampling)
    {
        passes[pass_count++] = resamplers[quality];
        passes[pass_count++] = pass_tag_slicer;
    }
    else if (source_format->block_size < target_format->block_size)
    {
        passes[pass_count++] = pass_tag_slicer;
    }

    for (int it = 0; it < sampling_rate_conversion.down_2x_pass_count; it++)
    {
        passes[pass_count++] = downsamplers_2x[quality][down_2x_index--];
    }

    if (target_format->layout != layout_interleaved)
    {
        if (target_format->channel_count > 1)
        {
            passes[pass_count++] = pass_tag_deinterleaver;
        }
    }

    ASSERT(pass_count <= max_pass_count);

    return pass_count;
}

static int resolve_formats
    (
        Format const * const source_format,
        Format const * const target_format,
        short const passes [const],
        int const pass_count,
        Format formats [const]
    )
{
    // Set the first format to the source format, the last format to the target
    // format, and clear the rest:

    formats[0] = *source_format;

    if (formats[0].channel_count == channel_count_1)
    {
        formats[0].layout = layout_interleaved;
    }

    for (int index = 1; index < pass_count; index++)
    {
        formats[index] = (Format){ 0 };
    }

    formats[pass_count] = *target_format;

    if (formats[pass_count].channel_count == channel_count_1)
    {
        formats[pass_count].layout = layout_interleaved;
    }

    // Do a few iterations:

    for (int it = 0; it < 2; it++)
    {
        // Forward pass:

        for (int index = 0; index < pass_count; index++)
        {
            Pass const pass = make_pass(passes[index]);

            pass.restrain_formats
            (
                pass.configuration,
                &formats[index],
                &formats[index + 1]
            );
        }

        // Backward pass:

        for (int index = pass_count - 1; index >= 0; index--)
        {
            Pass const pass = make_pass(passes[index]);

            pass.restrain_formats
            (
                pass.configuration,
                &formats[index],
                &formats[index + 1]
            );
        }
    }

    // TODO: Is this weird?

    if (pass_count == 0)
    {
        formats[0].max_block_count = max
        (
            source_format->max_block_count,
            target_format->max_block_count
        );
    }

    // The format count is always equal to the pass count plus one.

    return pass_count + 1;
}

static void resolve_conversion
    (
        Format const * const source_format,
        Format const * const target_format,
        DrB_Audio_Converter_Direction const direction,
        DrB_Audio_Converter_Quality const quality,
        DrB_Audio_Conversion * const conversion
    )
{
    switch (direction)
    {
      case drb_audio_converter_direction_push:

        conversion->pass_count = resolve_passes
        (
            source_format,
            target_format,
            quality,
            conversion->passes
        );

        conversion->format_count = resolve_formats
        (
            source_format,
            target_format,
            conversion->passes,
            conversion->pass_count,
            conversion->formats
        );

        break;

      case drb_audio_converter_direction_pull:

        conversion->pass_count = resolve_passes
        (
            target_format,
            source_format,
            quality,
            conversion->passes
        );

        conversion->format_count = resolve_formats
        (
            target_format,
            source_format,
            conversion->passes,
            conversion->pass_count,
            conversion->formats
        );

        invert_passes(conversion->passes, conversion->pass_count);

        break;
    }
}

static int required_buffer_size
    (
        Format const * const formats,
        int format_count
    )
{
    int length = 0;

    for (int index = 0; index < format_count; index++)
    {
        int const channel_count = formats[index].channel_count;
        int const block_size = formats[index].block_size;
        int const max_block_count = formats[index].max_block_count;

        length = max(length, channel_count * block_size * max_block_count);
    }

    return length * sizeof(float);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Validation

static bool any (bool const propositions [], int const count)
{
    bool acc = false;

    for (int index = 0; index < count; index++)
    {
        acc |= propositions[index];
    }

    return acc;
}

static bool all (bool const propositions [], int const count)
{
    bool acc = true;

    for (int index = 0; index < count; index++)
    {
        acc &= propositions[index];
    }

    return acc;
}

static bool validate_sampling_rate (int const sampling_rate)
{
    bool const propositions [] =
    {
        sampling_rate == drb_audio_converter_sampling_rate_8000,
        sampling_rate == drb_audio_converter_sampling_rate_11025,
        sampling_rate == drb_audio_converter_sampling_rate_16000,
        sampling_rate == drb_audio_converter_sampling_rate_22050,
        sampling_rate == drb_audio_converter_sampling_rate_44100,
        sampling_rate == drb_audio_converter_sampling_rate_48000,
        sampling_rate == drb_audio_converter_sampling_rate_60000,
        sampling_rate == drb_audio_converter_sampling_rate_88200,
        sampling_rate == drb_audio_converter_sampling_rate_96000,
        sampling_rate == drb_audio_converter_sampling_rate_120000,
        sampling_rate == drb_audio_converter_sampling_rate_176400,
        sampling_rate == drb_audio_converter_sampling_rate_192000,
        sampling_rate == drb_audio_converter_sampling_rate_240000
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_channel_count (int const channel_count)
{
    bool const propositions [] =
    {
        channel_count == channel_count_1,
        channel_count == channel_count_2,
        channel_count == channel_count_3,
        channel_count == channel_count_4,
        channel_count == channel_count_5,
        channel_count == channel_count_6,
        channel_count == channel_count_7,
        channel_count == channel_count_8
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_frame_layout (int const frame_layout)
{
    bool const propositions [] =
    {
        frame_layout == layout_interleaved,
        frame_layout == layout_deinterleaved
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_block_size (int const block_size)
{
    bool const propositions [] =
    {
        block_size == drb_audio_converter_block_size_1,
        block_size == drb_audio_converter_block_size_4,
        block_size == drb_audio_converter_block_size_16,
        block_size == drb_audio_converter_block_size_64,
        block_size == drb_audio_converter_block_size_256,
        block_size == drb_audio_converter_block_size_1024,
        block_size == drb_audio_converter_block_size_4096
    };

    return any(propositions, sizeof(propositions));
}

static bool validate_max_block_count(int const max_block_count)
{
    return 0 <= max_block_count && max_block_count <= 8192;
}

static bool validate_format (Format const * const format)
{
    bool const propositions [] =
    {
        validate_sampling_rate(format->sampling_rate),
        validate_channel_count(format->channel_count),
        validate_frame_layout(format->layout),
        validate_block_size(format->block_size),
        validate_max_block_count(format->max_block_count)
    };

    return all(propositions, sizeof(propositions));
}

static bool validate_conversion
    (
        Format const * const source_format,
        Format const * const target_format
    )
{
    bool const propositions [] =
    {
        validate_format(source_format),
        validate_format(target_format),
        source_format->block_size * source_format->max_block_count <= 8192,
        target_format->block_size * target_format->max_block_count <= 8192
    };

    return all(propositions, sizeof(propositions));
}

/* -------------------------------------------------------------------------- */

#pragma mark - Packing/Unpacking

static void pack_buffers_into_samples
    (
        DrB_Audio_Converter_Buffer const * const buffers,
        float * restrict const samples,
        int frame_count,
        int offset,
        int channel_count,
        int layout
    )
{
    switch (layout)
    {
      case layout_interleaved:
        memcpy
        (
            samples,
            buffers[0].samples + offset,
            frame_count * channel_count * sizeof(float)
        );
        break;
      case layout_deinterleaved:
        for (int channel = 0; channel < channel_count; channel++)
        {
            memcpy
            (
                samples + channel * frame_count,
                buffers[channel].samples + offset,
                frame_count * sizeof(float)
            );
        }
        break;
    }
}

static void unpack_samples_into_buffers
    (
        float const * restrict const samples,
        DrB_Audio_Converter_Buffer const * const buffers,
        int frame_count,
        int offset,
        int channel_count,
        int layout
    )
{
    switch (layout)
    {
      case layout_interleaved:
        memcpy
        (
            buffers[0].samples + offset,
            samples,
            frame_count * channel_count * sizeof(float)
        );
        break;
      case layout_deinterleaved:
        for (int channel = 0; channel < channel_count; channel++)
        {
            memcpy
            (
                buffers[channel].samples + offset,
                samples + channel * frame_count,
                frame_count * sizeof(float)
            );
        }
        break;
    }
}

static void make_buffers_point_to_samples
    (
        float * const samples,
        DrB_Audio_Converter_Buffer * const buffers,
        int const frame_count,
        int const channel_count
    )
{
    for (int channel = 0; channel < channel_count; channel++)
    {
        buffers[channel].samples = samples + channel * frame_count;
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Converter Definition

typedef void Converter_Function
    (
        struct DrB_Audio_Converter * converter,
        void * work_memory,
        double timestamp,
        DrB_Audio_Converter_Buffer const target_buffers [],
        int frame_count
    );

struct DrB_Audio_Converter
{
    Converter_Function * convert;
    DrB_Audio_Converter_Data_Callback data_callback;
    Format source_format, target_format;
    long buffer_length;
    int pass_count;
    int throttling;
    _Alignas(cache_line_size) Processor processors [];
};

/* -------------------------------------------------------------------------- */

#pragma mark - Conversion Push/Pull

static inline void swap_sample_pointers
    (
        float * restrict * restrict const buffer_a,
        float * restrict * restrict const buffer_b
    )
{
    float * const buffer_a_copy = *buffer_a;
    float * const buffer_b_copy = *buffer_b;

    *buffer_a = buffer_b_copy;
    *buffer_b = buffer_a_copy;
}

static void converter_push_frames
    (
        DrB_Audio_Converter * const converter,
        void * work_memory,
        double const timestamp,
        DrB_Audio_Converter_Buffer const source_buffers [const],
        int const frame_count
    )
{
    DrB_Audio_Converter_Buffer target_buffers [max_channel_count];

    int blocks [max_format_count];

    ASSERT(converter);

    float * source_samples = work_memory;
    float * target_samples = source_samples + converter->buffer_length;

    for (int offset = 0; offset < frame_count; offset += blocks[0])
    {
        double latency = 0.0;

        blocks[0] = min(frame_count - offset, converter->throttling);

        for (int index = 0; index < converter->pass_count; index++)
        {
            ASSERT(blocks[index] <= converter->buffer_length);

            Processor const processor = converter->processors[index];

            if (processor.pushed_target_frame_count)
            {
                blocks[index + 1] = processor.pushed_target_frame_count
                (
                    processor.state,
                    blocks[index],
                    &latency
                );
            }
            else
            {
                blocks[index + 1] = blocks[index];
            }
        }

        pack_buffers_into_samples
        (
            source_buffers,
            source_samples,
            blocks[0],
            offset,
            converter->source_format.channel_count,
            converter->source_format.layout
        );

        for (int index = 0; index < converter->pass_count; index++)
        {
            Processor const processor = converter->processors[index];

            int const processed_target_frame_count = processor.push
            (
                processor.state,
                source_samples,
                target_samples,
                blocks[index]
            );

            ASSERT(processed_target_frame_count == blocks[index + 1]);

            swap_sample_pointers(&source_samples, &target_samples);
        }

        make_buffers_point_to_samples
        (
            source_samples,
            target_buffers,
            blocks[converter->pass_count],
            converter->target_format.channel_count
        );

        converter->data_callback.process
        (
            converter->data_callback.state,
            timestamp + latency,
            target_buffers,
            blocks[converter->pass_count]
        );
    }
}

static void converter_pull_frames
    (
        DrB_Audio_Converter * const converter,
        void * work_buffer,
        double const timestamp,
        DrB_Audio_Converter_Buffer const target_buffers [const],
        int const frame_count
    )
{
    DrB_Audio_Converter_Buffer source_buffers [max_channel_count];

    int blocks [max_format_count];

    ASSERT(converter);

    float * source_samples = work_buffer;
    float * target_samples = source_samples + converter->buffer_length;

    for (int offset = 0; offset < frame_count; offset += blocks[0])
    {
        double latency = 0.0;

        blocks[0] = min(frame_count - offset, converter->throttling);

        for (int index = 0; index < converter->pass_count; index++)
        {
            ASSERT(blocks[index] <= converter->buffer_length);

            Processor const processor = converter->processors[index];

            if (processor.pulled_source_frame_count)
            {
                blocks[index + 1] = processor.pulled_source_frame_count
                (
                    processor.state,
                    blocks[index],
                    &latency
                );
            }
            else
            {
                blocks[index + 1] = blocks[index];
            }
        }

        make_buffers_point_to_samples
        (
            source_samples,
            source_buffers,
            blocks[converter->pass_count],
            converter->source_format.channel_count
        );

        converter->data_callback.process
        (
            converter->data_callback.state,
            timestamp + latency,
            source_buffers,
            blocks[converter->pass_count]
        );

        for (int index = converter->pass_count - 1; index >= 0; index--)
        {
            Processor const processor = converter->processors[index];

            int const processed_source_frame_count = processor.pull
            (
                processor.state,
                source_samples,
                target_samples,
                blocks[index]
            );

            ASSERT(processed_source_frame_count == blocks[index + 1]);

            swap_sample_pointers(&source_samples, &target_samples);
        }

        unpack_samples_into_buffers
        (
            source_samples,
            target_buffers,
            blocks[0],
            offset,
            converter->target_format.channel_count,
            converter->target_format.layout
        );
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Converter Construction

static void converter_construct
    (
        Bump_Allocator * const allocator,
        DrB_Audio_Conversion const * const conversion,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Data_Callback const data_callback
    )
{
    Format const * const formats = conversion->formats;
    short const * const passes = conversion->passes;
    int const pass_count = conversion->pass_count;
    int const format_count = conversion->format_count;

    DrB_Audio_Converter * const converter = alloc
    (
        allocator,
        sizeof(DrB_Audio_Converter) + pass_count * sizeof(Processor)
    );

    int const buffer_size = required_buffer_size(formats, format_count);
    int const throttling = formats[0].block_size * formats[0].max_block_count;

    if (converter)
    {
        switch (direction)
        {
          case drb_audio_converter_direction_push:
            converter->convert = converter_push_frames;
            converter->source_format = formats[0];
            converter->target_format = formats[pass_count];
            break;
          case drb_audio_converter_direction_pull:
            converter->convert = converter_pull_frames;
            converter->source_format = formats[pass_count];
            converter->target_format = formats[0];
            break;
        }

        converter->data_callback = data_callback;
        converter->buffer_length = cache_align(buffer_size) / sizeof(float);
        converter->pass_count = pass_count;
        converter->throttling = throttling;
    }

    for (int index = 0; index < pass_count; index++)
    {
        Pass const pass = make_pass(passes[index]);

        Processor processor;

        switch (direction)
        {
          case drb_audio_converter_direction_push:
            processor = pass.create_processor
            (
                pass.configuration,
                allocator,
                &formats[index + 0],
                &formats[index + 1]
            );
            break;
          case drb_audio_converter_direction_pull:
            processor = pass.create_processor
            (
                pass.configuration,
                allocator,
                &formats[index + 1],
                &formats[index + 0]
            );
            break;
        }

        if (converter)
        {
            converter->processors[index] = processor;
        }
    }
}

/* -------------------------------------------------------------------------- */

#pragma mark - Printing

#if defined(DEBUG) && defined(PRINT_CONVERSION)

#include <stdio.h>

static void print_format (Format const * const format)
{
    char const * const layout =
        format->layout == layout_interleaved
            ? "interleaved"
            : "de-interleaved";

    printf("sampling rate   = %d\n", format->sampling_rate);
    printf("channel count   = %d\n", format->channel_count);
    printf("layout          = %s\n", layout);
    printf("block size      = %d\n", format->block_size);
    printf("max block count = %d\n", format->max_block_count);
}

static void print_pass (int pass)
{
    static char const * const descriptions [] =
    {
        [pass_tag_interleaver] = "interleaver",
        [pass_tag_deinterleaver] = "deinterleaver",
        [pass_tag_upsampler_2x_fir_16] = "upsampler_2x_fir_16",
        [pass_tag_upsampler_2x_fir_32] = "upsampler_2x_fir_32",
        [pass_tag_upsampler_2x_fir_48] = "upsampler_2x_fir_48",
        [pass_tag_upsampler_2x_fir_64] = "upsampler_2x_fir_64",
        [pass_tag_upsampler_2x_fir_80] = "upsampler_2x_fir_80",
        [pass_tag_upsampler_2x_fir_96] = "upsampler_2x_fir_96",
        [pass_tag_downsampler_2x_fir_16] = "downsampler_2x_fir_16",
        [pass_tag_downsampler_2x_fir_32] = "downsampler_2x_fir_32",
        [pass_tag_downsampler_2x_fir_48] = "downsampler_2x_fir_48",
        [pass_tag_downsampler_2x_fir_64] = "downsampler_2x_fir_64",
        [pass_tag_downsampler_2x_fir_80] = "downsampler_2x_fir_80",
        [pass_tag_downsampler_2x_fir_96] = "downsampler_2x_fir_96",
        [pass_tag_resampler_sinc_8_linear] = "resampler_sinc_8_linear",
        [pass_tag_resampler_sinc_16_linear] = "resampler_sinc_16_linear",
        [pass_tag_resampler_sinc_24_linear] = "resampler_sinc_24_linear",
        [pass_tag_resampler_sinc_32_linear] = "resampler_sinc_32_linear",
        [pass_tag_resampler_sinc_8_cubic] = "resampler_sinc_8_cubic",
        [pass_tag_resampler_sinc_16_cubic] = "resampler_sinc_16_cubic",
        [pass_tag_resampler_sinc_24_cubic] = "resampler_sinc_24_cubic",
        [pass_tag_resampler_sinc_32_cubic] = "resampler_sinc_32_cubic",
        [pass_tag_slicer] = "slicer"
    };

    printf("%s\n", descriptions[pass]);
}

static void print_conversion (DrB_Audio_Conversion const * const conversion)
{
    printf("source\n\n");

    for (int index = 0; index < conversion->pass_count; index++)
    {
        print_format(&conversion->formats[index]);
        printf("\n");
        print_pass(conversion->passes[index]);
        printf("\n");
    }

    print_format(&conversion->formats[conversion->pass_count]);
    printf("\n");
    printf("target\n");
    printf("\n");
}

#endif // defined(DEBUG) && defined(PRINT_CONVERSION)

/* -------------------------------------------------------------------------- */

#pragma mark - Converter API Functions

extern _Bool drb_audio_converter_alignment_and_size
    (
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int block_size,
        int max_block_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality quality,
        long * alignment,
        long * size
    )
{
    int const source_block_size =
        direction == drb_audio_converter_direction_pull ? block_size : 1;

    int const target_block_size =
        direction == drb_audio_converter_direction_push ? block_size : 1;

    int const source_max_block_count =
        direction == drb_audio_converter_direction_pull ? max_block_count : 0;

    int const target_max_block_count =
        direction == drb_audio_converter_direction_push ? max_block_count : 0;

    Format const source_format =
    {
        .sampling_rate = source_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = source_block_size,
        .max_block_count = source_max_block_count
    };

    Format const target_format =
    {
        .sampling_rate = target_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = target_block_size,
        .max_block_count = target_max_block_count
    };

    // Check that the conversion is valid.

    if (!validate_conversion(&source_format, &target_format))
    {
        return 0;
    }

    // Resolve the conversion.

    DrB_Audio_Conversion conversion = { 0 };

    resolve_conversion
    (
        &source_format,
        &target_format,
        direction,
        quality,
        &conversion
    );

    // Set up an empty allocator in order to estimate the required memory size.

    DrB_Audio_Converter_Data_Callback const data_callback = { 0 };

    Bump_Allocator bump_allocator;

    bump_allocator_construct(&bump_allocator, 0);

    converter_construct(&bump_allocator, &conversion, direction, data_callback);

    *alignment = cache_line_size;
    *size = bump_allocator.offset;

    return true;
}

extern DrB_Audio_Converter * drb_audio_converter_construct
    (
        void * memory,
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int block_size,
        int max_block_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality quality,
        DrB_Audio_Converter_Data_Callback data_callback
    )
{
    int const source_block_size =
        direction == drb_audio_converter_direction_pull ? block_size : 1;

    int const target_block_size =
        direction == drb_audio_converter_direction_push ? block_size : 1;

    int const source_max_block_count =
        direction == drb_audio_converter_direction_pull ? max_block_count : 0;

    int const target_max_block_count =
        direction == drb_audio_converter_direction_push ? max_block_count : 0;

    Format const source_format =
    {
        .sampling_rate = source_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = source_block_size,
        .max_block_count = source_max_block_count
    };

    Format const target_format =
    {
        .sampling_rate = target_sampling_rate,
        .channel_count = channel_count,
        .layout = layout_deinterleaved,
        .block_size = target_block_size,
        .max_block_count = target_max_block_count
    };

    // Check that the conversion is valid.

    if (!validate_conversion(&source_format, &target_format))
    {
        return 0;
    }

    // Resolve the conversion.

    DrB_Audio_Conversion conversion = { 0 };

    resolve_conversion
    (
        &source_format,
        &target_format,
        direction,
        quality,
        &conversion
    );

  #if defined(DEBUG) && defined(PRINT_CONVERSION)
    print_conversion(&conversion);
  #endif

    Bump_Allocator bump_allocator;

    bump_allocator_construct(&bump_allocator, memory);

    converter_construct(&bump_allocator, &conversion, direction, data_callback);

    // Cast the pointer to the memory implicitly to a converter and return it.

    return memory;
}

extern void drb_audio_converter_work_memory_alignment_and_size
    (
        DrB_Audio_Converter * converter,
        long * alignment,
        long * size
    )
{
    *alignment = cache_line_size;
    *size = converter->buffer_length * 2 * sizeof(float);
}

extern void drb_audio_converter_process
    (
        DrB_Audio_Converter * const converter,
        void * const work_memory,
        double const timestamp,
        DrB_Audio_Converter_Buffer const buffers [const],
        int const frame_count
    )
{
    converter->convert(converter, work_memory, timestamp, buffers, frame_count);
}

/* -------------------------------------------------------------------------- */

#pragma mark - Kernels

_Alignas(64) static float const kernel_2x_fir_16 [8] =
{
    -0.0001199071f, +0.0050233615f, -0.0454816414f, +0.2905768050f,
    +0.2905768050f, -0.0454816414f, +0.0050233615f, -0.0001199071f
};

_Alignas(64) static float const kernel_2x_fir_32 [16] =
{
    -0.0000108295f, +0.0001999758f, -0.0012197609f, +0.0047545301f,
    -0.0141238117f, +0.0355866439f, -0.0863425372f, +0.3111579909f,
    +0.3111579909f, -0.0863425372f, +0.0355866439f, -0.0141238117f,
    +0.0047545301f, -0.0012197609f, +0.0001999758f, -0.0000108295f
};

_Alignas(64) static float const kernel_2x_fir_48 [24] =
{
    -0.0000029696f, +0.0000399690f, -0.0001883739f, +0.0006249117f,
    -0.0016744538f, +0.0038602660f, -0.0079648422f, +0.0151605471f,
    -0.0274429195f, +0.0493349697f, -0.0968589350f, +0.3151128616f,
    +0.3151128616f, -0.0968589350f, +0.0493349697f, -0.0274429195f,
    +0.0151605471f, -0.0079648422f, +0.0038602660f, -0.0016744538f,
    +0.0006249117f, -0.0001883739f, +0.0000399690f, -0.0000029696f
};

_Alignas(64) static float const kernel_2x_fir_64 [32] =
{
    -0.0000012146f, +0.0000141345f, -0.0000568089f, +0.0001666104f,
    -0.0004088859f, +0.0008855596f, -0.0017440518f, +0.0031867119f,
    -0.0054838807f, +0.0090004334f, -0.0142617463f, +0.0221301322f,
    -0.0343195801f, +0.0551936171f, -0.1008084555f, +0.3165079191f,
    +0.3165079191f, -0.1008084555f, +0.0551936171f, -0.0343195801f,
    +0.0221301322f, -0.0142617463f, +0.0090004334f, -0.0054838807f,
    +0.0031867119f, -0.0017440518f, +0.0008855596f, -0.0004088859f,
    +0.0001666104f, -0.0000568089f, +0.0000141345f, -0.0000012146f
};

_Alignas(64) static float const kernel_2x_fir_80 [40] =
{
    -0.0000006121f, +0.0000065923f, -0.0000239814f, +0.0000642740f,
    -0.0001468086f, +0.0003008955f, -0.0005681180f, +0.0010046723f,
    -0.0016837438f, +0.0026982588f, -0.0041650018f, +0.0062323398f,
    -0.0090963283f, +0.0130356862f, -0.0184908195f, +0.0262557595f,
    -0.0380076767f, +0.0581153610f, -0.1026860286f, +0.3171555468f,
    +0.3171555468f, -0.1026860286f, +0.0581153610f, -0.0380076767f,
    +0.0262557595f, -0.0184908195f, +0.0130356862f, -0.0090963283f,
    +0.0062323398f, -0.0041650018f, +0.0026982588f, -0.0016837438f,
    +0.0010046723f, -0.0005681180f, +0.0003008955f, -0.0001468086f,
    +0.0000642740f, -0.0000239814f, +0.0000065923f, -0.0000006121f
};

_Alignas(64) static float const kernel_2x_fir_96 [48] =
{
    -0.0000003509f, +0.0000036098f, -0.0000123073f, +0.0000308976f,
    -0.0000666586f, +0.0001303586f, -0.0002369979f, +0.0004065870f,
    -0.0006649309f, +0.0010444142f, -0.0015848434f, +0.0023345053f,
    -0.0033517786f, +0.0047079372f, -0.0064923410f, +0.0088222963f,
    -0.0118622146f, +0.0158622292f, -0.0212409738f, +0.0287808457f,
    -0.0401598844f, +0.0597612362f, -0.1037193303f, +0.3175078537f,
    +0.3175078537f, -0.1037193303f, +0.0597612362f, -0.0401598844f,
    +0.0287808457f, -0.0212409738f, +0.0158622292f, -0.0118622146f,
    +0.0088222963f, -0.0064923410f, +0.0047079372f, -0.0033517786f,
    +0.0023345053f, -0.0015848434f, +0.0010444142f, -0.0006649309f,
    +0.0004065870f, -0.0002369979f, +0.0001303586f, -0.0000666586f,
    +0.0000308976f, -0.0000123073f, +0.0000036098f, -0.0000003509f
};

_Alignas(64) static float const kernels_sinc_8_linear [64][8] =
{
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0001000957f, +0.0016211868f, -0.0106713673f, +0.9994941258f,
        +0.0112808321f, -0.0017367647f, +0.0001121543f, -0.0000000074f
    },
    {
        -0.0001887210f, +0.0031277163f, -0.0207338006f, +0.9979776224f,
        +0.0231701337f, -0.0035895990f, +0.0002369367f, -0.0000000594f
    },
    {
        -0.0002664782f, +0.0045209111f, -0.0301893169f, +0.9954538433f,
        +0.0356653475f, -0.0055585397f, +0.0003748925f, -0.0000002015f
    },
    {
        -0.0003339790f, +0.0058024688f, -0.0390413762f, +0.9919283670f,
        +0.0487623231f, -0.0076431427f, +0.0005265380f, -0.0000004795f
    },
    {
        -0.0003918400f, +0.0069744349f, -0.0472948336f, +0.9874089790f,
        +0.0624552898f, -0.0098424583f, +0.0006923540f, -0.0000009404f
    },
    {
        -0.0004406791f, +0.0080391750f, -0.0549558892f, +0.9819056492f,
        +0.0767368351f, -0.0121550071f, +0.0008727809f, -0.0000016318f
    },
    {
        -0.0004811114f, +0.0089993480f, -0.0620320336f, +0.9754305010f,
        +0.0915978865f, -0.0145787568f, +0.0010682119f, -0.0000026022f
    },
    {
        -0.0005137457f, +0.0098578786f, -0.0685319915f, +0.9679977745f,
        +0.1070277002f, -0.0171111005f, +0.0012789872f, -0.0000039005f
    },
    {
        -0.0005391820f, +0.0106179306f, -0.0744656623f, +0.9596237841f,
        +0.1230138533f, -0.0197488358f, +0.0015053876f, -0.0000055765f
    },
    {
        -0.0005580079f, +0.0112828800f, -0.0798440584f, +0.9503268686f,
        +0.1395422422f, -0.0224881461f, +0.0017476276f, -0.0000076806f
    },
    {
        -0.0005707964f, +0.0118562897f, -0.0846792411f, +0.9401273359f,
        +0.1565970863f, -0.0253245824f, +0.0020058495f, -0.0000102634f
    },
    {
        -0.0005781039f, +0.0123418829f, -0.0889842551f, +0.9290474023f,
        +0.1741609370f, -0.0282530479f, +0.0022801160f, -0.0000133759f
    },
    {
        -0.0005804676f, +0.0127435192f, -0.0927730617f, +0.9171111257f,
        +0.1922146919f, -0.0312677838f, +0.0025704041f, -0.0000170690f
    },
    {
        -0.0005784045f, +0.0130651700f, -0.0960604696f, +0.9043443332f,
        +0.2107376154f, -0.0343623570f, +0.0028765981f, -0.0000213935f
    },
    {
        -0.0005724092f, +0.0133108948f, -0.0988620664f, +0.8907745442f,
        +0.2297073643f, -0.0375296506f, +0.0031984829f, -0.0000263993f
    },
    {
        -0.0005629535f, +0.0134848193f, -0.1011941480f, +0.8764308888f,
        +0.2491000189f, -0.0407618558f, +0.0035357379f, -0.0000321356f
    },
    {
        -0.0005504849f, +0.0135911134f, -0.1030736479f, +0.8613440209f,
        +0.2688901198f, -0.0440504665f, +0.0038879300f, -0.0000386500f
    },
    {
        -0.0005354261f, +0.0136339708f, -0.1045180672f, +0.8455460278f,
        +0.2890507108f, -0.0473862767f, +0.0042545077f, -0.0000459884f
    },
    {
        -0.0005181746f, +0.0136175888f, -0.1055454034f, +0.8290703355f,
        +0.3095533862f, -0.0507593799f, +0.0046347953f, -0.0000541939f
    },
    {
        -0.0004991021f, +0.0135461506f, -0.1061740803f, +0.8119516113f,
        +0.3303683443f, -0.0541591714f, +0.0050279865f, -0.0000633067f
    },
    {
        -0.0004785548f, +0.0134238068f, -0.1064228789f, +0.7942256621f,
        +0.3514644458f, -0.0575743532f, +0.0054331395f, -0.0000733632f
    },
    {
        -0.0004568531f, +0.0132546598f, -0.1063108675f, +0.7759293303f,
        +0.3728092774f, -0.0609929423f, +0.0058491718f, -0.0000843952f
    },
    {
        -0.0004342923f, +0.0130427481f, -0.1058573353f, +0.7571003880f,
        +0.3943692207f, -0.0644022812f, +0.0062748555f, -0.0000964292f
    },
    {
        -0.0004111426f, +0.0127920323f, -0.1050817248f, +0.7377774281f,
        +0.4161095250f, -0.0677890514f, +0.0067088130f, -0.0001094855f
    },
    {
        -0.0003876498f, +0.0125063818f, -0.1040035676f, +0.7179997542f,
        +0.4379943860f, -0.0711392906f, +0.0071495137f, -0.0001235773f
    },
    {
        -0.0003640359f, +0.0121895635f, -0.1026424209f, +0.6978072692f,
        +0.4599870277f, -0.0744384123f, +0.0075952705f, -0.0001387098f
    },
    {
        -0.0003405001f, +0.0118452306f, -0.1010178060f, +0.6772403630f,
        +0.4820497896f, -0.0776712285f, +0.0080442375f, -0.0001548789f
    },
    {
        -0.0003172193f, +0.0114769133f, -0.0991491492f, +0.6563397995f,
        +0.5041442166f, -0.0808219758f, +0.0084944082f, -0.0001720706f
    },
    {
        -0.0002943492f, +0.0110880102f, -0.0970557246f, +0.6351466037f,
        +0.5262311537f, -0.0838743445f, +0.0089436146f, -0.0001902595f
    },
    {
        -0.0002720256f, +0.0106817810f, -0.0947565995f, +0.6137019490f,
        +0.5482708434f, -0.0868115106f, +0.0093895266f, -0.0002094080f
    },
    {
        -0.0002503647f, +0.0102613404f, -0.0922705819f, +0.5920470450f,
        +0.5702230263f, -0.0896161710f, +0.0098296524f, -0.0002294650f
    },
    {
        -0.0002294650f, +0.0098296524f, -0.0896161710f, +0.5702230263f,
        +0.5920470450f, -0.0922705819f, +0.0102613404f, -0.0002503647f
    },
    {
        -0.0002094080f, +0.0093895266f, -0.0868115106f, +0.5482708434f,
        +0.6137019490f, -0.0947565995f, +0.0106817810f, -0.0002720256f
    },
    {
        -0.0001902595f, +0.0089436146f, -0.0838743445f, +0.5262311537f,
        +0.6351466037f, -0.0970557246f, +0.0110880102f, -0.0002943492f
    },
    {
        -0.0001720706f, +0.0084944082f, -0.0808219758f, +0.5041442166f,
        +0.6563397995f, -0.0991491492f, +0.0114769133f, -0.0003172193f
    },
    {
        -0.0001548789f, +0.0080442375f, -0.0776712285f, +0.4820497896f,
        +0.6772403630f, -0.1010178060f, +0.0118452306f, -0.0003405001f
    },
    {
        -0.0001387098f, +0.0075952705f, -0.0744384123f, +0.4599870277f,
        +0.6978072692f, -0.1026424209f, +0.0121895635f, -0.0003640359f
    },
    {
        -0.0001235773f, +0.0071495137f, -0.0711392906f, +0.4379943860f,
        +0.7179997542f, -0.1040035676f, +0.0125063818f, -0.0003876498f
    },
    {
        -0.0001094855f, +0.0067088130f, -0.0677890514f, +0.4161095250f,
        +0.7377774281f, -0.1050817248f, +0.0127920323f, -0.0004111426f
    },
    {
        -0.0000964292f, +0.0062748555f, -0.0644022812f, +0.3943692207f,
        +0.7571003880f, -0.1058573353f, +0.0130427481f, -0.0004342923f
    },
    {
        -0.0000843952f, +0.0058491718f, -0.0609929423f, +0.3728092774f,
        +0.7759293303f, -0.1063108675f, +0.0132546598f, -0.0004568531f
    },
    {
        -0.0000733632f, +0.0054331395f, -0.0575743532f, +0.3514644458f,
        +0.7942256621f, -0.1064228789f, +0.0134238068f, -0.0004785548f
    },
    {
        -0.0000633067f, +0.0050279865f, -0.0541591714f, +0.3303683443f,
        +0.8119516113f, -0.1061740803f, +0.0135461506f, -0.0004991021f
    },
    {
        -0.0000541939f, +0.0046347953f, -0.0507593799f, +0.3095533862f,
        +0.8290703355f, -0.1055454034f, +0.0136175888f, -0.0005181746f
    },
    {
        -0.0000459884f, +0.0042545077f, -0.0473862767f, +0.2890507108f,
        +0.8455460278f, -0.1045180672f, +0.0136339708f, -0.0005354261f
    },
    {
        -0.0000386500f, +0.0038879300f, -0.0440504665f, +0.2688901198f,
        +0.8613440209f, -0.1030736479f, +0.0135911134f, -0.0005504849f
    },
    {
        -0.0000321356f, +0.0035357379f, -0.0407618558f, +0.2491000189f,
        +0.8764308888f, -0.1011941480f, +0.0134848193f, -0.0005629535f
    },
    {
        -0.0000263993f, +0.0031984829f, -0.0375296506f, +0.2297073643f,
        +0.8907745442f, -0.0988620664f, +0.0133108948f, -0.0005724092f
    },
    {
        -0.0000213935f, +0.0028765981f, -0.0343623570f, +0.2107376154f,
        +0.9043443332f, -0.0960604696f, +0.0130651700f, -0.0005784045f
    },
    {
        -0.0000170690f, +0.0025704041f, -0.0312677838f, +0.1922146919f,
        +0.9171111257f, -0.0927730617f, +0.0127435192f, -0.0005804676f
    },
    {
        -0.0000133759f, +0.0022801160f, -0.0282530479f, +0.1741609370f,
        +0.9290474023f, -0.0889842551f, +0.0123418829f, -0.0005781039f
    },
    {
        -0.0000102634f, +0.0020058495f, -0.0253245824f, +0.1565970863f,
        +0.9401273359f, -0.0846792411f, +0.0118562897f, -0.0005707964f
    },
    {
        -0.0000076806f, +0.0017476276f, -0.0224881461f, +0.1395422422f,
        +0.9503268686f, -0.0798440584f, +0.0112828800f, -0.0005580079f
    },
    {
        -0.0000055765f, +0.0015053876f, -0.0197488358f, +0.1230138533f,
        +0.9596237841f, -0.0744656623f, +0.0106179306f, -0.0005391820f
    },
    {
        -0.0000039005f, +0.0012789872f, -0.0171111005f, +0.1070277002f,
        +0.9679977745f, -0.0685319915f, +0.0098578786f, -0.0005137457f
    },
    {
        -0.0000026022f, +0.0010682119f, -0.0145787568f, +0.0915978865f,
        +0.9754305010f, -0.0620320336f, +0.0089993480f, -0.0004811114f
    },
    {
        -0.0000016318f, +0.0008727809f, -0.0121550071f, +0.0767368351f,
        +0.9819056492f, -0.0549558892f, +0.0080391750f, -0.0004406791f
    },
    {
        -0.0000009404f, +0.0006923540f, -0.0098424583f, +0.0624552898f,
        +0.9874089790f, -0.0472948336f, +0.0069744349f, -0.0003918400f
    },
    {
        -0.0000004795f, +0.0005265380f, -0.0076431427f, +0.0487623231f,
        +0.9919283670f, -0.0390413762f, +0.0058024688f, -0.0003339790f
    },
    {
        -0.0000002015f, +0.0003748925f, -0.0055585397f, +0.0356653475f,
        +0.9954538433f, -0.0301893169f, +0.0045209111f, -0.0002664782f
    },
    {
        -0.0000000594f, +0.0002369367f, -0.0035895990f, +0.0231701337f,
        +0.9979776224f, -0.0207338006f, +0.0031277163f, -0.0001887210f
    },
    {
        -0.0000000074f, +0.0001121543f, -0.0017367647f, +0.0112808321f,
        +0.9994941258f, -0.0106713673f, +0.0016211868f, -0.0001000957f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    }
};

_Alignas(64) static float const kernels_sinc_16_linear [64][16] =
{
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000057207f, +0.0000515028f, -0.0002450600f, +0.0008247364f,
        -0.0022341931f, +0.0054101868f, -0.0142160545f, +0.9995627353f,
        +0.0147601424f, -0.0055625295f, +0.0023003694f, -0.0008536286f,
        +0.0002558223f, -0.0000545168f, +0.0000062418f, -0.0000000009f
    },
    {
        -0.0000109325f, +0.0000999712f, -0.0004790426f, +0.0016191715f,
        -0.0043979774f, +0.0106581020f, -0.0278768303f, +0.9982516759f,
        +0.0300519950f, -0.0112668092f, +0.0046623742f, -0.0017346057f,
        +0.0005220434f, -0.0001120149f, +0.0000130150f, -0.0000000074f
    },
    {
        -0.0000156491f, +0.0001453971f, -0.0007016802f, +0.0023820308f,
        -0.0064874323f, +0.0157344888f, -0.0409723314f, +0.9960690246f,
        +0.0458620041f, -0.0171015935f, +0.0070811695f, -0.0026412537f,
        +0.0007982505f, -0.0001724487f, +0.0000203285f, -0.0000000250f
    },
    {
        -0.0000198854f, +0.0001877832f, -0.0009127518f, +0.0031121739f,
        -0.0084989519f, +0.0206307696f, -0.0534937625f, +0.9930184474f,
        +0.0621754537f, -0.0230549961f, +0.0095516125f, -0.0035717615f,
        +0.0010839815f, -0.0002357593f, +0.0000281888f, -0.0000000592f
    },
    {
        -0.0000236576f, +0.0002271435f, -0.0011120822f, +0.0038085940f,
        -0.0104292467f, +0.0253390511f, -0.0654335311f, +0.9891050659f,
        +0.0789764817f, -0.0291145044f, +0.0120682684f, -0.0045241863f,
        +0.0013787243f, -0.0003018738f, +0.0000366005f, -0.0000001154f
    },
    {
        -0.0000269830f, +0.0002635019f, -0.0012995403f, +0.0044704162f,
        -0.0122753442f, +0.0298521272f, -0.0767852481f, +0.9843354464f,
        +0.0962480987f, -0.0352669951f, +0.0146254186f, -0.0054964554f,
        +0.0016819163f, -0.0003707053f, +0.0000455658f, -0.0000001992f
    },
    {
        -0.0000298797f, +0.0002968921f, -0.0014750385f, +0.0050968967f,
        -0.0140345888f, +0.0341634806f, -0.0875437245f, +0.9787175859f,
        +0.1139722101f, -0.0414987523f, +0.0172170691f, -0.0064863679f,
        +0.0019929451f, -0.0004421522f, +0.0000550846f, -0.0000003157f
    },
    {
        -0.0000323668f, +0.0003273570f, -0.0016385307f, +0.0056874203f,
        -0.0157046408f, +0.0382672828f, -0.0977049670f, +0.9722608952f,
        +0.1321296399f, -0.0477954875f, +0.0198369600f, -0.0074915982f,
        +0.0023111478f, -0.0005160980f, +0.0000651539f, -0.0000004700f
    },
    {
        -0.0000344638f, +0.0003549477f, -0.0017900112f, +0.0062414984f,
        -0.0172834743f, +0.0421583919f, -0.1072661694f, +0.9649761792f,
        +0.1507001581f, -0.0541423604f, +0.0224785765f, -0.0085096984f,
        +0.0026358118f, -0.0005924111f, +0.0000757684f, -0.0000006670f
    },
    {
        -0.0000361908f, +0.0003797233f, -0.0019295131f, +0.0067587664f,
        -0.0187693746f, +0.0458323490f, -0.1162257027f, +0.9568756138f,
        +0.1696625098f, -0.0605240028f, +0.0251351601f, -0.0095381028f,
        +0.0029661751f, -0.0006709445f, +0.0000869195f, -0.0000009113f
    },
    {
        -0.0000375682f, +0.0004017501f, -0.0020571066f, +0.0072389811f,
        -0.0201609344f, +0.0492853729f, -0.1245831019f, +0.9479727201f,
        +0.1889944475f, -0.0669245429f, +0.0277997213f, -0.0105741312f,
        +0.0033014271f, -0.0007515355f, +0.0000985957f, -0.0000012074f
    },
    {
        -0.0000386169f, +0.0004211010f, -0.0021728973f, +0.0076820170f,
        -0.0214570497f, +0.0525143532f, -0.1323390510f, +0.9382823363f,
        +0.2086727654f, -0.0733276328f, +0.0304650537f, -0.0116149942f,
        +0.0036407091f, -0.0008340057f, +0.0001107823f, -0.0000015593f
    },
    {
        -0.0000393577f, +0.0004378551f, -0.0022770247f, +0.0080878635f,
        -0.0226569144f, +0.0555168421f, -0.1394953648f, +0.9278205858f,
        +0.2286733366f, -0.0797164760f, +0.0331237479f, -0.0126577981f,
        +0.0039831156f, -0.0009181608f, +0.0001234613f, -0.0000019708f
    },
    {
        -0.0000398116f, +0.0004520968f, -0.0023696600f, +0.0084566209f,
        -0.0237600144f, +0.0582910442f, -0.1460549696f, +0.9166048439f,
        +0.2489711518f, -0.0860738581f, +0.0357682074f, -0.0136995502f,
        +0.0043276958f, -0.0010037908f, +0.0001366113f, -0.0000024452f
    },
    {
        -0.0000399996f, +0.0004639154f, -0.0024510046f, +0.0087884965f,
        -0.0247661215f, +0.0608358053f, -0.1520218801f, +0.9046537007f,
        +0.2695403613f, -0.0923821776f, +0.0383906650f, -0.0147371653f,
        +0.0046734542f, -0.0010906698f, +0.0001502073f, -0.0000029855f
    },
    {
        -0.0000399422f, +0.0004734045f, -0.0025212883f, +0.0090838006f,
        -0.0256752857f, +0.0631505999f, -0.1574011759f, +0.8919869224f,
        +0.2903543185f, -0.0986234798f, +0.0409832001f, -0.0157674721f,
        +0.0050193531f, -0.0011785563f, +0.0001642206f, -0.0000035942f
    },
    {
        -0.0000396601f, +0.0004806613f, -0.0025807671f, +0.0093429424f,
        -0.0264878277f, +0.0652355167f, -0.1621989743f, +0.8786254098f,
        +0.3113856254f, -0.1047794904f, +0.0435377568f, -0.0167872198f,
        +0.0053643141f, -0.0012671929f, +0.0001786188f, -0.0000042732f
    },
    {
        -0.0000391734f, +0.0004857862f, -0.0026297215f, +0.0095664249f,
        -0.0272043304f, +0.0670912435f, -0.1664224022f, +0.8645911544f,
        +0.3326061808f, -0.1108316521f, +0.0460461631f, -0.0177930863f,
        +0.0057072200f, -0.0013563073f, +0.0001933658f, -0.0000050242f
    },
    {
        -0.0000385020f, +0.0004888823f, -0.0026684545f, +0.0097548411f,
        -0.0278256303f, +0.0687190510f, -0.1700795655f, +0.8499071927f,
        +0.3539872298f, -0.1167611615f, +0.0485001508f, -0.0187816853f,
        +0.0060469171f, -0.0014456118f, +0.0002084213f, -0.0000058481f
    },
    {
        -0.0000376651f, +0.0004900546f, -0.0026972898f, +0.0099088687f,
        -0.0283528076f, +0.0701207746f, -0.1731795171f, +0.8345975577f,
        +0.3754994153f, -0.1225490075f, +0.0508913757f, -0.0197495751f,
        +0.0063822179f, -0.0015348043f, +0.0002237412f, -0.0000067452f
    },
    {
        -0.0000366816f, +0.0004894097f, -0.0027165698f, +0.0100292653f,
        -0.0287871766f, +0.0712987962f, -0.1757322229f, +0.8186872296f,
        +0.3971128310f, -0.1281760111f, +0.0532114394f, -0.0206932671f,
        +0.0067119034f, -0.0016235681f, +0.0002392774f, -0.0000077154f
    },
    {
        -0.0000355696f, +0.0004870554f, -0.0027266535f, +0.0101168639f,
        -0.0291302756f, +0.0722560243f, -0.1777485262f, +0.8022020833f,
        +0.4187970765f, -0.1336228659f, +0.0554519112f, -0.0216092353f,
        +0.0070347262f, -0.0017115731f, +0.0002549777f, -0.0000087576f
    },
    {
        -0.0000343468f, +0.0004831000f, -0.0027279153f, +0.0101725669f,
        -0.0293838561f, +0.0729958732f, -0.1792401109f, +0.7851688349f,
        +0.4405213132f, -0.1388701801f, +0.0576043506f, -0.0224939250f,
        +0.0073494134f, -0.0017984758f, +0.0002707860f, -0.0000098705f
    },
    {
        -0.0000330300f, +0.0004776519f, -0.0027207421f, +0.0101973422f,
        -0.0295498716f, +0.0735222419f, -0.1802194629f, +0.7676149870f,
        +0.4622543220f, -0.1438985191f, +0.0596603305f, -0.0233437637f,
        +0.0076546697f, -0.0018839203f, +0.0002866419f, -0.0000110517f
    },
    {
        -0.0000316353f, +0.0004708193f, -0.0027055324f, +0.0101922171f,
        -0.0296304668f, +0.0738394917f, -0.1806998304f, +0.7495687713f,
        +0.4839645622f, -0.1486884486f, +0.0616114613f, -0.0241551705f,
        +0.0079491813f, -0.0019675388f, +0.0003024811f, -0.0000122982f
    },
    {
        -0.0000301782f, +0.0004627096f, -0.0026826940f, +0.0101582734f,
        -0.0296279658f, +0.0739524232f, -0.1806951832f, +0.7310590910f,
        +0.5056202312f, -0.1532205793f, +0.0634494148f, -0.0249245669f,
        +0.0082316189f, -0.0020489527f, +0.0003182355f, -0.0000136063f
    },
    {
        -0.0000286733f, +0.0004534294f, -0.0026526422f, +0.0100966428f,
        -0.0295448603f, +0.0738662530f, -0.1802201704f, +0.7121154618f,
        +0.5271893253f, -0.1574756111f, +0.0651659490f, -0.0256483879f,
        +0.0085006417f, -0.0021277731f, +0.0003338327f, -0.0000149714f
    },
    {
        -0.0000271343f, +0.0004430835f, -0.0026157984f, +0.0100085008f,
        -0.0293837979f, +0.0735865890f, -0.1792900778f, +0.6927679511f,
        +0.5486397016f, -0.1614343793f, +0.0667529331f, -0.0263230927f,
        +0.0087549017f, -0.0022036020f, +0.0003491969f, -0.0000163883f
    },
    {
        -0.0000255742f, +0.0004317752f, -0.0025725882f, +0.0098950624f,
        -0.0291475698f, +0.0731194065f, -0.1779207841f, +0.6730471171f,
        +0.5699391399f, -0.1650778992f, +0.0682023730f, -0.0269451762f,
        +0.0089930473f, -0.0022760333f, +0.0003642481f, -0.0000178509f
    },
    {
        -0.0000240051f, +0.0004196056f, -0.0025234398f, +0.0097575763f,
        -0.0288390992f, +0.0724710229f, -0.1761287163f, +0.6529839470f,
        +0.5910554062f, -0.1683874134f, +0.0695064367f, -0.0275111806f,
        +0.0092137276f, -0.0023446539f, +0.0003789030f, -0.0000193520f
    },
    {
        -0.0000224383f, +0.0004066735f, -0.0024687824f, +0.0095973206f,
        -0.0284614285f, +0.0716480722f, -0.1739308046f, +0.6326097942f,
        +0.6119563151f, -0.1713444370f, +0.0706574799f, -0.0280177073f,
        +0.0094155973f, -0.0024090448f, +0.0003930749f, -0.0000208841f
    },
    {
        -0.0000208841f, +0.0003930749f, -0.0024090448f, +0.0094155973f,
        -0.0280177073f, +0.0706574799f, -0.1713444370f, +0.6119563151f,
        +0.6326097942f, -0.1739308046f, +0.0716480722f, -0.0284614285f,
        +0.0095973206f, -0.0024687824f, +0.0004066735f, -0.0000224383f
    },
    {
        -0.0000193520f, +0.0003789030f, -0.0023446539f, +0.0092137276f,
        -0.0275111806f, +0.0695064367f, -0.1683874134f, +0.5910554062f,
        +0.6529839470f, -0.1761287163f, +0.0724710229f, -0.0288390992f,
        +0.0097575763f, -0.0025234398f, +0.0004196056f, -0.0000240051f
    },
    {
        -0.0000178509f, +0.0003642481f, -0.0022760333f, +0.0089930473f,
        -0.0269451762f, +0.0682023730f, -0.1650778992f, +0.5699391399f,
        +0.6730471171f, -0.1779207841f, +0.0731194065f, -0.0291475698f,
        +0.0098950624f, -0.0025725882f, +0.0004317752f, -0.0000255742f
    },
    {
        -0.0000163883f, +0.0003491969f, -0.0022036020f, +0.0087549017f,
        -0.0263230927f, +0.0667529331f, -0.1614343793f, +0.5486397016f,
        +0.6927679511f, -0.1792900778f, +0.0735865890f, -0.0293837979f,
        +0.0100085008f, -0.0026157984f, +0.0004430835f, -0.0000271343f
    },
    {
        -0.0000149714f, +0.0003338327f, -0.0021277731f, +0.0085006417f,
        -0.0256483879f, +0.0651659490f, -0.1574756111f, +0.5271893253f,
        +0.7121154618f, -0.1802201704f, +0.0738662530f, -0.0295448603f,
        +0.0100966428f, -0.0026526422f, +0.0004534294f, -0.0000286733f
    },
    {
        -0.0000136063f, +0.0003182355f, -0.0020489527f, +0.0082316189f,
        -0.0249245669f, +0.0634494148f, -0.1532205793f, +0.5056202312f,
        +0.7310590910f, -0.1806951832f, +0.0739524232f, -0.0296279658f,
        +0.0101582734f, -0.0026826940f, +0.0004627096f, -0.0000301782f
    },
    {
        -0.0000122982f, +0.0003024811f, -0.0019675388f, +0.0079491813f,
        -0.0241551705f, +0.0616114613f, -0.1486884486f, +0.4839645622f,
        +0.7495687713f, -0.1806998304f, +0.0738394917f, -0.0296304668f,
        +0.0101922171f, -0.0027055324f, +0.0004708193f, -0.0000316353f
    },
    {
        -0.0000110517f, +0.0002866419f, -0.0018839203f, +0.0076546697f,
        -0.0233437637f, +0.0596603305f, -0.1438985191f, +0.4622543220f,
        +0.7676149870f, -0.1802194629f, +0.0735222419f, -0.0295498716f,
        +0.0101973422f, -0.0027207421f, +0.0004776519f, -0.0000330300f
    },
    {
        -0.0000098705f, +0.0002707860f, -0.0017984758f, +0.0073494134f,
        -0.0224939250f, +0.0576043506f, -0.1388701801f, +0.4405213132f,
        +0.7851688349f, -0.1792401109f, +0.0729958732f, -0.0293838561f,
        +0.0101725669f, -0.0027279153f, +0.0004831000f, -0.0000343468f
    },
    {
        -0.0000087576f, +0.0002549777f, -0.0017115731f, +0.0070347262f,
        -0.0216092353f, +0.0554519112f, -0.1336228659f, +0.4187970765f,
        +0.8022020833f, -0.1777485262f, +0.0722560243f, -0.0291302756f,
        +0.0101168639f, -0.0027266535f, +0.0004870554f, -0.0000355696f
    },
    {
        -0.0000077154f, +0.0002392774f, -0.0016235681f, +0.0067119034f,
        -0.0206932671f, +0.0532114394f, -0.1281760111f, +0.3971128310f,
        +0.8186872296f, -0.1757322229f, +0.0712987962f, -0.0287871766f,
        +0.0100292653f, -0.0027165698f, +0.0004894097f, -0.0000366816f
    },
    {
        -0.0000067452f, +0.0002237412f, -0.0015348043f, +0.0063822179f,
        -0.0197495751f, +0.0508913757f, -0.1225490075f, +0.3754994153f,
        +0.8345975577f, -0.1731795171f, +0.0701207746f, -0.0283528076f,
        +0.0099088687f, -0.0026972898f, +0.0004900546f, -0.0000376651f
    },
    {
        -0.0000058481f, +0.0002084213f, -0.0014456118f, +0.0060469171f,
        -0.0187816853f, +0.0485001508f, -0.1167611615f, +0.3539872298f,
        +0.8499071927f, -0.1700795655f, +0.0687190510f, -0.0278256303f,
        +0.0097548411f, -0.0026684545f, +0.0004888823f, -0.0000385020f
    },
    {
        -0.0000050242f, +0.0001933658f, -0.0013563073f, +0.0057072200f,
        -0.0177930863f, +0.0460461631f, -0.1108316521f, +0.3326061808f,
        +0.8645911544f, -0.1664224022f, +0.0670912435f, -0.0272043304f,
        +0.0095664249f, -0.0026297215f, +0.0004857862f, -0.0000391734f
    },
    {
        -0.0000042732f, +0.0001786188f, -0.0012671929f, +0.0053643141f,
        -0.0167872198f, +0.0435377568f, -0.1047794904f, +0.3113856254f,
        +0.8786254098f, -0.1621989743f, +0.0652355167f, -0.0264878277f,
        +0.0093429424f, -0.0025807671f, +0.0004806613f, -0.0000396601f
    },
    {
        -0.0000035942f, +0.0001642206f, -0.0011785563f, +0.0050193531f,
        -0.0157674721f, +0.0409832001f, -0.0986234798f, +0.2903543185f,
        +0.8919869224f, -0.1574011759f, +0.0631505999f, -0.0256752857f,
        +0.0090838006f, -0.0025212883f, +0.0004734045f, -0.0000399422f
    },
    {
        -0.0000029855f, +0.0001502073f, -0.0010906698f, +0.0046734542f,
        -0.0147371653f, +0.0383906650f, -0.0923821776f, +0.2695403613f,
        +0.9046537007f, -0.1520218801f, +0.0608358053f, -0.0247661215f,
        +0.0087884965f, -0.0024510046f, +0.0004639154f, -0.0000399996f
    },
    {
        -0.0000024452f, +0.0001366113f, -0.0010037908f, +0.0043276958f,
        -0.0136995502f, +0.0357682074f, -0.0860738581f, +0.2489711518f,
        +0.9166048439f, -0.1460549696f, +0.0582910442f, -0.0237600144f,
        +0.0084566209f, -0.0023696600f, +0.0004520968f, -0.0000398116f
    },
    {
        -0.0000019708f, +0.0001234613f, -0.0009181608f, +0.0039831156f,
        -0.0126577981f, +0.0331237479f, -0.0797164760f, +0.2286733366f,
        +0.9278205858f, -0.1394953648f, +0.0555168421f, -0.0226569144f,
        +0.0080878635f, -0.0022770247f, +0.0004378551f, -0.0000393577f
    },
    {
        -0.0000015593f, +0.0001107823f, -0.0008340057f, +0.0036407091f,
        -0.0116149942f, +0.0304650537f, -0.0733276328f, +0.2086727654f,
        +0.9382823363f, -0.1323390510f, +0.0525143532f, -0.0214570497f,
        +0.0076820170f, -0.0021728973f, +0.0004211010f, -0.0000386169f
    },
    {
        -0.0000012074f, +0.0000985957f, -0.0007515355f, +0.0033014271f,
        -0.0105741312f, +0.0277997213f, -0.0669245429f, +0.1889944475f,
        +0.9479727201f, -0.1245831019f, +0.0492853729f, -0.0201609344f,
        +0.0072389811f, -0.0020571066f, +0.0004017501f, -0.0000375682f
    },
    {
        -0.0000009113f, +0.0000869195f, -0.0006709445f, +0.0029661751f,
        -0.0095381028f, +0.0251351601f, -0.0605240028f, +0.1696625098f,
        +0.9568756138f, -0.1162257027f, +0.0458323490f, -0.0187693746f,
        +0.0067587664f, -0.0019295131f, +0.0003797233f, -0.0000361908f
    },
    {
        -0.0000006670f, +0.0000757684f, -0.0005924111f, +0.0026358118f,
        -0.0085096984f, +0.0224785765f, -0.0541423604f, +0.1507001581f,
        +0.9649761792f, -0.1072661694f, +0.0421583919f, -0.0172834743f,
        +0.0062414984f, -0.0017900112f, +0.0003549477f, -0.0000344638f
    },
    {
        -0.0000004700f, +0.0000651539f, -0.0005160980f, +0.0023111478f,
        -0.0074915982f, +0.0198369600f, -0.0477954875f, +0.1321296399f,
        +0.9722608952f, -0.0977049670f, +0.0382672828f, -0.0157046408f,
        +0.0056874203f, -0.0016385307f, +0.0003273570f, -0.0000323668f
    },
    {
        -0.0000003157f, +0.0000550846f, -0.0004421522f, +0.0019929451f,
        -0.0064863679f, +0.0172170691f, -0.0414987523f, +0.1139722101f,
        +0.9787175859f, -0.0875437245f, +0.0341634806f, -0.0140345888f,
        +0.0050968967f, -0.0014750385f, +0.0002968921f, -0.0000298797f
    },
    {
        -0.0000001992f, +0.0000455658f, -0.0003707053f, +0.0016819163f,
        -0.0054964554f, +0.0146254186f, -0.0352669951f, +0.0962480987f,
        +0.9843354464f, -0.0767852481f, +0.0298521272f, -0.0122753442f,
        +0.0044704162f, -0.0012995403f, +0.0002635019f, -0.0000269830f
    },
    {
        -0.0000001154f, +0.0000366005f, -0.0003018738f, +0.0013787243f,
        -0.0045241863f, +0.0120682684f, -0.0291145044f, +0.0789764817f,
        +0.9891050659f, -0.0654335311f, +0.0253390511f, -0.0104292467f,
        +0.0038085940f, -0.0011120822f, +0.0002271435f, -0.0000236576f
    },
    {
        -0.0000000592f, +0.0000281888f, -0.0002357593f, +0.0010839815f,
        -0.0035717615f, +0.0095516125f, -0.0230549961f, +0.0621754537f,
        +0.9930184474f, -0.0534937625f, +0.0206307696f, -0.0084989519f,
        +0.0031121739f, -0.0009127518f, +0.0001877832f, -0.0000198854f
    },
    {
        -0.0000000250f, +0.0000203285f, -0.0001724487f, +0.0007982505f,
        -0.0026412537f, +0.0070811695f, -0.0171015935f, +0.0458620041f,
        +0.9960690246f, -0.0409723314f, +0.0157344888f, -0.0064874323f,
        +0.0023820308f, -0.0007016802f, +0.0001453971f, -0.0000156491f
    },
    {
        -0.0000000074f, +0.0000130150f, -0.0001120149f, +0.0005220434f,
        -0.0017346057f, +0.0046623742f, -0.0112668092f, +0.0300519950f,
        +0.9982516759f, -0.0278768303f, +0.0106581020f, -0.0043979774f,
        +0.0016191715f, -0.0004790426f, +0.0000999712f, -0.0000109325f
    },
    {
        -0.0000000009f, +0.0000062418f, -0.0000545168f, +0.0002558223f,
        -0.0008536286f, +0.0023003694f, -0.0055625295f, +0.0147601424f,
        +0.9995627353f, -0.0142160545f, +0.0054101868f, -0.0022341931f,
        +0.0008247364f, -0.0002450600f, +0.0000515028f, -0.0000057207f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    }
};

_Alignas(64) static float const kernels_sinc_24_linear [64][24] =
{
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000013543f, +0.0000089600f, -0.0000346636f, +0.0001026222f,
        -0.0002541553f, +0.0005529969f, -0.0010933876f, +0.0020226998f,
        -0.0036235103f, +0.0066725664f, -0.0149802521f, +0.9995754412f,
        +0.0155032976f, -0.0068146398f, +0.0036912163f, -0.0020610560f,
        +0.0011159162f, -0.0005658377f, +0.0002609715f, -0.0001058757f,
        +0.0000360031f, -0.0000094071f, +0.0000014621f, -0.0000000003f
    },
    {
        -0.0000026019f, +0.0000174642f, -0.0000679341f, +0.0002018017f,
        -0.0005009808f, +0.0010919740f, -0.0021618434f, +0.0040025479f,
        -0.0071713821f, +0.0131896874f, -0.0294241423f, +0.9983024341f,
        +0.0315152454f, -0.0137573690f, +0.0074418877f, -0.0041557880f,
        +0.0022518491f, -0.0011432751f, +0.0005282128f, -0.0002148003f,
        +0.0000732860f, -0.0000192507f, +0.0000030327f, -0.0000000022f
    },
    {
        -0.0000037441f, +0.0000255057f, -0.0000997641f, +0.0002973648f,
        -0.0007399959f, +0.0016158161f, -0.0032030780f, +0.0059352416f,
        -0.0106360150f, +0.0195386219f, -0.0433194477f, +0.9961829853f,
        +0.0480203864f, -0.0208136120f, +0.0112434608f, -0.0062793402f,
        +0.0034051820f, -0.0017310111f, +0.0008011455f, -0.0003265541f,
        +0.0001117828f, -0.0000295179f, +0.0000047117f, -0.0000000074f
    },
    {
        -0.0000047827f, +0.0000330795f, -0.0001301121f, +0.0003891534f,
        -0.0009707542f, +0.0021234736f, -0.0042149188f, +0.0078166789f,
        -0.0140101545f, +0.0257072773f, -0.0566550514f, +0.9932204346f,
        +0.0650022208f, -0.0279682243f, +0.0150870956f, -0.0084266909f,
        +0.0045731990f, -0.0023276870f, +0.0010791604f, -0.0004409024f,
        +0.0001514213f, -0.0000401940f, +0.0000064989f, -0.0000000175f
    },
    {
        -0.0000057198f, +0.0000401823f, -0.0001589427f, +0.0004770246f,
        -0.0011928432f, +0.0026139641f, -0.0051953143f, +0.0096429655f,
        -0.0172869036f, +0.0316842250f, -0.0694209524f, +0.9894194491f,
        +0.0824432278f, -0.0352055163f, +0.0189636830f, -0.0105926626f,
        +0.0057530904f, -0.0029318889f, +0.0013616178f, -0.0005575957f,
        +0.0001921235f, -0.0000512623f, +0.0000083935f, -0.0000000340f
    },
    {
        -0.0000065579f, +0.0000468126f, -0.0001862264f, +0.0005608515f,
        -0.0014058851f, +0.0030863730f, -0.0061423367f, +0.0114104213f,
        -0.0204597341f, +0.0374587147f, -0.0816082736f, +0.9847860134f,
        +0.1003248892f, -0.0425092785f, +0.0228638630f, -0.0127719331f,
        +0.0069419578f, -0.0035421504f, +0.0016478494f, -0.0006763702f,
        +0.0002338048f, -0.0000627037f, +0.0000103942f, -0.0000000587f
    },
    {
        -0.0000072998f, +0.0000529707f, -0.0002119393f, +0.0006405222f,
        -0.0016095368f, +0.0035398554f, -0.0070541848f, +0.0131155858f,
        -0.0235224967f, +0.0430206862f, -0.0932092669f, +0.9793274188f,
        +0.1186277152f, -0.0498628084f, +0.0267780441f, -0.0149590466f,
        +0.0081368213f, -0.0041569558f, +0.0019371588f, -0.0007969480f,
        +0.0002763747f, -0.0000744974f, +0.0000124989f, -0.0000000928f
    },
    {
        -0.0000079485f, +0.0000586582f, -0.0002360633f, +0.0007159403f,
        -0.0018034901f, +0.0039736359f, -0.0079291864f, +0.0147552236f,
        -0.0264694294f, +0.0483607803f, -0.1042173167f, +0.9730522488f,
        +0.1373312726f, -0.0572489392f, +0.0306964231f, -0.0171484256f,
        +0.0093346247f, -0.0047747428f, +0.0022288231f, -0.0009190376f,
        +0.0003197369f, -0.0000866200f, +0.0000147054f, -0.0000001380f
    },
    {
        -0.0000085072f, +0.0000638784f, -0.0002585856f, +0.0007870246f,
        -0.0019874714f, +0.0043870101f, -0.0087657999f, +0.0163263284f,
        -0.0292951658f, +0.0534703475f, -0.1146269404f, +0.9659703625f,
        +0.1564142147f, -0.0646500691f, +0.0346090067f, -0.0193343838f,
        +0.0105322435f, -0.0053939061f, +0.0025220943f, -0.0010423343f,
        +0.0003637889f, -0.0000990464f, +0.0000170105f, -0.0000001955f
    },
    {
        -0.0000089795f, +0.0000686360f, -0.0002794990f, +0.0008537086f,
        -0.0021612420f, +0.0047793444f, -0.0095626163f, +0.0178261270f,
        -0.0319947412f, +0.0583414553f, -0.1244337870f, +0.9580928759f,
        +0.1758543135f, -0.0720481930f, +0.0385056324f, -0.0215111387f,
        +0.0117264913f, -0.0060128009f, +0.0028162008f, -0.0011665209f,
        +0.0004084226f, -0.0001117489f, +0.0000194108f, -0.0000002666f
    },
    {
        -0.0000093689f, +0.0000729370f, -0.0002988014f, +0.0009159407f,
        -0.0023245977f, +0.0051500762f, -0.0103183601f, +0.0192520818f,
        -0.0345635983f, +0.0629668935f, -0.1336346333f, +0.9494321403f,
        +0.1956284938f, -0.0794249346f, +0.0423759922f, -0.0236728252f,
        +0.0129141272f, -0.0066297465f, +0.0031103489f, -0.0012912681f,
        +0.0004535241f, -0.0001246981f, +0.0000219020f, -0.0000003526f
    },
    {
        -0.0000096794f, +0.0000767889f, -0.0003164955f, +0.0009736836f,
        -0.0024773684f, +0.0054987142f, -0.0110318900f, +0.0206018939f,
        -0.0369975917f, +0.0673401775f, -0.1422273781f, +0.9400017183f,
        +0.2157128693f, -0.0867615800f, +0.0462096549f, -0.0258135098f,
        +0.0140918639f, -0.0072430303f, +0.0034037249f, -0.0014162353f,
        +0.0004989741f, -0.0001378621f, +0.0000244792f, -0.0000004544f
    },
    {
        -0.0000099150f, +0.0000802001f, -0.0003325890f, +0.0010269141f,
        -0.0026194179f, +0.0058248378f, -0.0117021996f, +0.0218735040f,
        -0.0392929910f, +0.0714555512f, -0.1502110337f, +0.9298163582f,
        +0.2360827804f, -0.0940391131f, +0.0499960908f, -0.0279272044f,
        +0.0152563753f, -0.0078509119f, +0.0036954963f, -0.0015410710f,
        +0.0005446482f, -0.0001512071f, +0.0000271371f, -0.0000005731f
    },
    {
        -0.0000100797f, +0.0000831804f, -0.0003470944f, +0.0010756229f,
        -0.0027506433f, +0.0061280969f, -0.0123284168f, +0.0230650942f,
        -0.0414464831f, +0.0753079865f, -0.1575857157f, +0.9188919652f,
        +0.2567128338f, -0.1012382511f, +0.0537246961f, -0.0300078815f,
        +0.0164043051f, -0.0084516270f, +0.0039848144f, -0.0016654138f,
        +0.0005904165f, -0.0001646973f, +0.0000298696f, -0.0000007094f
    },
    {
        -0.0000101778f, +0.0000857404f, -0.0003600283f, +0.0011198137f,
        -0.0028709744f, +0.0064082111f, -0.0129098044f, +0.0241750876f,
        -0.0434551734f, +0.0788931824f, -0.1643526302f, +0.9072455711f,
        +0.2775769441f, -0.1083394817f, +0.0573848185f, -0.0320494892f,
        +0.0175322749f, -0.0090433922f, +0.0042708158f, -0.0017888932f,
        +0.0006361448f, -0.0001782946f, +0.0000326699f, -0.0000008641f
    },
    {
        -0.0000102134f, +0.0000878918f, -0.0003714116f, +0.0011595035f,
        -0.0029803734f, +0.0066649691f, -0.0134457585f, +0.0252021486f,
        -0.0453165853f, +0.0822075621f, -0.1705140594f, +0.8948953014f,
        +0.2986483765f, -0.1153231015f, +0.0609657826f, -0.0340459666f,
        +0.0186368932f, -0.0096244094f, +0.0045526249f, -0.0019111303f,
        +0.0006816939f, -0.0001919592f, +0.0000355307f, -0.0000010376f
    },
    {
        -0.0000101911f, +0.0000896472f, -0.0003812695f, +0.0011947217f,
        -0.0030788336f, +0.0068982276f, -0.0139358081f, +0.0261451817f,
        -0.0470286601f, +0.0852482682f, -0.1760733446f, +0.8818603408f,
        +0.3198997917f, -0.1221692546f, +0.0644569162f, -0.0359912595f,
        +0.0197147640f, -0.0101928703f, +0.0048293557f, -0.0020317388f,
        +0.0007269205f, -0.0002056490f, +0.0000384440f, -0.0000012305f
    },
    {
        -0.0000101150f, +0.0000910200f, -0.0003896305f, +0.0012255096f,
        -0.0031663791f, +0.0071079104f, -0.0143796131f, +0.0270033302f,
        -0.0485897541f, +0.0880131567f, -0.1810348678f, +0.8681608965f,
        +0.3413032914f, -0.1288579723f, +0.0678475774f, -0.0378793365f,
        +0.0207624964f, -0.0107469611f, +0.0051001145f, -0.0021503259f,
        +0.0007716772f, -0.0002193205f, +0.0000414011f, -0.0000014428f
    },
    {
        -0.0000099895f, +0.0000920241f, -0.0003965270f, +0.0012519203f,
        -0.0032430638f, +0.0072940067f, -0.0147769632f, +0.0277759740f,
        -0.0499986360f, +0.0905007894f, -0.1854040306f, +0.8538181592f,
        +0.3628304664f, -0.1353692140f, +0.0711271815f, -0.0397042050f,
        +0.0217767131f, -0.0112848681f, +0.0053640019f, -0.0022664930f,
        +0.0008158131f, -0.0002329281f, +0.0000443926f, -0.0000016747f
    },
    {
        -0.0000098190f, +0.0000926744f, -0.0004019947f, +0.0012740178f,
        -0.0033089704f, +0.0074565700f, -0.0151277757f, +0.0284627267f,
        -0.0512544827f, +0.0927104247f, -0.1891872322f, +0.8388542632f,
        +0.3844524447f, -0.1416829083f, +0.0742852284f, -0.0414599280f,
        +0.0227540605f, -0.0118047819f, +0.0056201154f, -0.0023798373f,
        +0.0008591740f, -0.0002464244f, +0.0000474087f, -0.0000019259f
    },
    {
        -0.0000096078f, +0.0000929860f, -0.0004060722f, +0.0012918767f,
        -0.0033642094f, +0.0075957164f, -0.0154320928f, +0.0290634328f,
        -0.0523568743f, +0.0946420071f, -0.1923918444f, +0.8232922436f,
        +0.4061399413f, -0.1477789946f, +0.0773113302f, -0.0431406402f,
        +0.0236912176f, -0.0123049027f, +0.0058675520f, -0.0024899524f,
        +0.0009016025f, -0.0002597607f, +0.0000504388f, -0.0000021963f
    },
    {
        -0.0000093600f, +0.0000929747f, -0.0004088012f, +0.0013055816f,
        -0.0034089182f, +0.0077116226f, -0.0156900794f, +0.0295781635f,
        -0.0533057882f, +0.0962961549f, -0.1950261859f, +0.8071559931f,
        +0.4278633099f, -0.1536374655f, +0.0801952389f, -0.0447405648f,
        +0.0245849060f, -0.0127834460f, +0.0061054104f, -0.0025964297f,
        +0.0009429389f, -0.0002728866f, +0.0000534716f, -0.0000024852f
    },
    {
        -0.0000090800f, +0.0000926567f, -0.0004102258f, +0.0013152265f,
        -0.0034432600f, +0.0078045247f, -0.0159020199f, +0.0300072123f,
        -0.0541015922f, +0.0976741471f, -0.1970994944f, +0.7904702161f,
        +0.4495925938f, -0.1592384096f, +0.0829268748f, -0.0462540303f,
        +0.0254318994f, -0.0132386471f, +0.0063327940f, -0.0026988592f,
        +0.0009830214f, -0.0002857503f, +0.0000564953f, -0.0000027920f
    },
    {
        -0.0000087717f, +0.0000920484f, -0.0004103927f, +0.0013209143f,
        -0.0034674225f, +0.0078747154f, -0.0160683154f, +0.0303510906f,
        -0.0547450369f, +0.0987779087f, -0.1986218968f, +0.7732603822f,
        +0.4712975796f, -0.1645620543f, +0.0854963540f, -0.0476754869f,
        +0.0262290333f, -0.0136687671f, +0.0065488136f, -0.0027968310f,
        +0.0010216864f, -0.0002982986f, +0.0000594976f, -0.0000031158f
    },
    {
        -0.0000084391f, +0.0000911665f, -0.0004093506f, +0.0013227563f,
        -0.0034816169f, +0.0079225425f, -0.0161894798f, +0.0306105220f,
        -0.0552372469f, +0.0996099944f, -0.1996043789f, +0.7555526778f,
        +0.4929478504f, -0.1695888094f, +0.0878940166f, -0.0489995235f,
        +0.0269732144f, -0.0140720976f, +0.0067525894f, -0.0028899364f,
        +0.0010587691f, -0.0003104775f, +0.0000624655f, -0.0000034554f
    },
    {
        -0.0000080861f, +0.0000900279f, -0.0004071501f, +0.0013208713f,
        -0.0034860766f, +0.0079484064f, -0.0162661365f, +0.0307864365f,
        -0.0555797115f, +0.1001735716f, -0.2000587521f, +0.7373739569f,
        +0.5145128398f, -0.1742993098f, +0.0901104547f, -0.0502208841f,
        +0.0276614309f, -0.0144469667f, +0.0069432549f, -0.0029777692f,
        +0.0010941039f, -0.0003222315f, +0.0000653854f, -0.0000038096f
    },
    {
        -0.0000077165f, +0.0000886497f, -0.0004038436f, +0.0013153853f,
        -0.0034810559f, +0.0079527578f, -0.0162990143f, +0.0308799646f,
        -0.0557742748f, +0.1004724024f, -0.1999976205f, +0.7187516901f,
        +0.5359618872f, -0.1786744596f, +0.0921365398f, -0.0513344843f,
        +0.0282907616f, -0.0147917438f, +0.0071199584f, -0.0030599267f,
        +0.0011275250f, -0.0003335047f, +0.0000682434f, -0.0000041767f
    },
    {
        -0.0000073337f, +0.0000870488f, -0.0003994851f, +0.0013064309f,
        -0.0034668289f, +0.0079360955f, -0.0162889434f, +0.0308924300f,
        -0.0558231246f, +0.1005108235f, -0.1994343451f, +0.6997139134f,
        +0.5572642930f, -0.1826954750f, +0.0939634505f, -0.0523354279f,
        +0.0288583858f, -0.0151048454f, +0.0072818669f, -0.0031360114f,
        +0.0011588670f, -0.0003442401f, +0.0000710249f, -0.0000045552f
    },
    {
        -0.0000069414f, +0.0000852424f, -0.0003941294f, +0.0012941467f,
        -0.0034436881f, +0.0078989637f, -0.0162368510f, +0.0308253433f,
        -0.0557287807f, +0.1002937264f, -0.1983830079f, +0.6802891755f,
        +0.5783893738f, -0.1863439276f, +0.0955827000f, -0.0532190223f,
        +0.0293615926f, -0.0153847404f, +0.0074281681f, -0.0032056319f,
        +0.0011879648f, -0.0003543804f, +0.0000737149f, -0.0000049431f
    },
    {
        -0.0000065426f, +0.0000832474f, -0.0003878327f, +0.0012786764f,
        -0.0034119430f, +0.0078419496f, -0.0161437571f, +0.0306803942f,
        -0.0554940832f, +0.0998265352f, -0.1968583742f, +0.6605064842f,
        +0.5993065186f, -0.1896017874f, +0.0969861622f, -0.0539807952f,
        +0.0297977904f, -0.0156299551f, +0.0075580737f, -0.0032684048f,
        +0.0012146551f, -0.0003638678f, +0.0000762982f, -0.0000053382f
    },
    {
        -0.0000061406f, +0.0000810806f, -0.0003806518f, +0.0012601687f,
        -0.0033719188f, +0.0077656807f, -0.0160107694f, +0.0304594436f,
        -0.0551221792f, +0.0991151852f, -0.1948758546f, +0.6403952526f,
        +0.6199852444f, -0.1924514651f, +0.0981660989f, -0.0546165096f,
        +0.0301645159f, -0.0158390790f, +0.0076708220f, -0.0033239552f,
        +0.0012387761f, -0.0003726442f, +0.0000787588f, -0.0000057382f
    },
    {
        -0.0000057382f, +0.0000787588f, -0.0003726442f, +0.0012387761f,
        -0.0033239552f, +0.0076708220f, -0.0158390790f, +0.0301645159f,
        -0.0546165096f, +0.0981660989f, -0.1924514651f, +0.6199852444f,
        +0.6403952526f, -0.1948758546f, +0.0991151852f, -0.0551221792f,
        +0.0304594436f, -0.0160107694f, +0.0077656807f, -0.0033719188f,
        +0.0012601687f, -0.0003806518f, +0.0000810806f, -0.0000061406f
    },
    {
        -0.0000053382f, +0.0000762982f, -0.0003638678f, +0.0012146551f,
        -0.0032684048f, +0.0075580737f, -0.0156299551f, +0.0297977904f,
        -0.0539807952f, +0.0969861622f, -0.1896017874f, +0.5993065186f,
        +0.6605064842f, -0.1968583742f, +0.0998265352f, -0.0554940832f,
        +0.0306803942f, -0.0161437571f, +0.0078419496f, -0.0034119430f,
        +0.0012786764f, -0.0003878327f, +0.0000832474f, -0.0000065426f
    },
    {
        -0.0000049431f, +0.0000737149f, -0.0003543804f, +0.0011879648f,
        -0.0032056319f, +0.0074281681f, -0.0153847404f, +0.0293615926f,
        -0.0532190223f, +0.0955827000f, -0.1863439276f, +0.5783893738f,
        +0.6802891755f, -0.1983830079f, +0.1002937264f, -0.0557287807f,
        +0.0308253433f, -0.0162368510f, +0.0078989637f, -0.0034436881f,
        +0.0012941467f, -0.0003941294f, +0.0000852424f, -0.0000069414f
    },
    {
        -0.0000045552f, +0.0000710249f, -0.0003442401f, +0.0011588670f,
        -0.0031360114f, +0.0072818669f, -0.0151048454f, +0.0288583858f,
        -0.0523354279f, +0.0939634505f, -0.1826954750f, +0.5572642930f,
        +0.6997139134f, -0.1994343451f, +0.1005108235f, -0.0558231246f,
        +0.0308924300f, -0.0162889434f, +0.0079360955f, -0.0034668289f,
        +0.0013064309f, -0.0003994851f, +0.0000870488f, -0.0000073337f
    },
    {
        -0.0000041767f, +0.0000682434f, -0.0003335047f, +0.0011275250f,
        -0.0030599267f, +0.0071199584f, -0.0147917438f, +0.0282907616f,
        -0.0513344843f, +0.0921365398f, -0.1786744596f, +0.5359618872f,
        +0.7187516901f, -0.1999976205f, +0.1004724024f, -0.0557742748f,
        +0.0308799646f, -0.0162990143f, +0.0079527578f, -0.0034810559f,
        +0.0013153853f, -0.0004038436f, +0.0000886497f, -0.0000077165f
    },
    {
        -0.0000038096f, +0.0000653854f, -0.0003222315f, +0.0010941039f,
        -0.0029777692f, +0.0069432549f, -0.0144469667f, +0.0276614309f,
        -0.0502208841f, +0.0901104547f, -0.1742993098f, +0.5145128398f,
        +0.7373739569f, -0.2000587521f, +0.1001735716f, -0.0555797115f,
        +0.0307864365f, -0.0162661365f, +0.0079484064f, -0.0034860766f,
        +0.0013208713f, -0.0004071501f, +0.0000900279f, -0.0000080861f
    },
    {
        -0.0000034554f, +0.0000624655f, -0.0003104775f, +0.0010587691f,
        -0.0028899364f, +0.0067525894f, -0.0140720976f, +0.0269732144f,
        -0.0489995235f, +0.0878940166f, -0.1695888094f, +0.4929478504f,
        +0.7555526778f, -0.1996043789f, +0.0996099944f, -0.0552372469f,
        +0.0306105220f, -0.0161894798f, +0.0079225425f, -0.0034816169f,
        +0.0013227563f, -0.0004093506f, +0.0000911665f, -0.0000084391f
    },
    {
        -0.0000031158f, +0.0000594976f, -0.0002982986f, +0.0010216864f,
        -0.0027968310f, +0.0065488136f, -0.0136687671f, +0.0262290333f,
        -0.0476754869f, +0.0854963540f, -0.1645620543f, +0.4712975796f,
        +0.7732603822f, -0.1986218968f, +0.0987779087f, -0.0547450369f,
        +0.0303510906f, -0.0160683154f, +0.0078747154f, -0.0034674225f,
        +0.0013209143f, -0.0004103927f, +0.0000920484f, -0.0000087717f
    },
    {
        -0.0000027920f, +0.0000564953f, -0.0002857503f, +0.0009830214f,
        -0.0026988592f, +0.0063327940f, -0.0132386471f, +0.0254318994f,
        -0.0462540303f, +0.0829268748f, -0.1592384096f, +0.4495925938f,
        +0.7904702161f, -0.1970994944f, +0.0976741471f, -0.0541015922f,
        +0.0300072123f, -0.0159020199f, +0.0078045247f, -0.0034432600f,
        +0.0013152265f, -0.0004102258f, +0.0000926567f, -0.0000090800f
    },
    {
        -0.0000024852f, +0.0000534716f, -0.0002728866f, +0.0009429389f,
        -0.0025964297f, +0.0061054104f, -0.0127834460f, +0.0245849060f,
        -0.0447405648f, +0.0801952389f, -0.1536374655f, +0.4278633099f,
        +0.8071559931f, -0.1950261859f, +0.0962961549f, -0.0533057882f,
        +0.0295781635f, -0.0156900794f, +0.0077116226f, -0.0034089182f,
        +0.0013055816f, -0.0004088012f, +0.0000929747f, -0.0000093600f
    },
    {
        -0.0000021963f, +0.0000504388f, -0.0002597607f, +0.0009016025f,
        -0.0024899524f, +0.0058675520f, -0.0123049027f, +0.0236912176f,
        -0.0431406402f, +0.0773113302f, -0.1477789946f, +0.4061399413f,
        +0.8232922436f, -0.1923918444f, +0.0946420071f, -0.0523568743f,
        +0.0290634328f, -0.0154320928f, +0.0075957164f, -0.0033642094f,
        +0.0012918767f, -0.0004060722f, +0.0000929860f, -0.0000096078f
    },
    {
        -0.0000019259f, +0.0000474087f, -0.0002464244f, +0.0008591740f,
        -0.0023798373f, +0.0056201154f, -0.0118047819f, +0.0227540605f,
        -0.0414599280f, +0.0742852284f, -0.1416829083f, +0.3844524447f,
        +0.8388542632f, -0.1891872322f, +0.0927104247f, -0.0512544827f,
        +0.0284627267f, -0.0151277757f, +0.0074565700f, -0.0033089704f,
        +0.0012740178f, -0.0004019947f, +0.0000926744f, -0.0000098190f
    },
    {
        -0.0000016747f, +0.0000443926f, -0.0002329281f, +0.0008158131f,
        -0.0022664930f, +0.0053640019f, -0.0112848681f, +0.0217767131f,
        -0.0397042050f, +0.0711271815f, -0.1353692140f, +0.3628304664f,
        +0.8538181592f, -0.1854040306f, +0.0905007894f, -0.0499986360f,
        +0.0277759740f, -0.0147769632f, +0.0072940067f, -0.0032430638f,
        +0.0012519203f, -0.0003965270f, +0.0000920241f, -0.0000099895f
    },
    {
        -0.0000014428f, +0.0000414011f, -0.0002193205f, +0.0007716772f,
        -0.0021503259f, +0.0051001145f, -0.0107469611f, +0.0207624964f,
        -0.0378793365f, +0.0678475774f, -0.1288579723f, +0.3413032914f,
        +0.8681608965f, -0.1810348678f, +0.0880131567f, -0.0485897541f,
        +0.0270033302f, -0.0143796131f, +0.0071079104f, -0.0031663791f,
        +0.0012255096f, -0.0003896305f, +0.0000910200f, -0.0000101150f
    },
    {
        -0.0000012305f, +0.0000384440f, -0.0002056490f, +0.0007269205f,
        -0.0020317388f, +0.0048293557f, -0.0101928703f, +0.0197147640f,
        -0.0359912595f, +0.0644569162f, -0.1221692546f, +0.3198997917f,
        +0.8818603408f, -0.1760733446f, +0.0852482682f, -0.0470286601f,
        +0.0261451817f, -0.0139358081f, +0.0068982276f, -0.0030788336f,
        +0.0011947217f, -0.0003812695f, +0.0000896472f, -0.0000101911f
    },
    {
        -0.0000010376f, +0.0000355307f, -0.0001919592f, +0.0006816939f,
        -0.0019111303f, +0.0045526249f, -0.0096244094f, +0.0186368932f,
        -0.0340459666f, +0.0609657826f, -0.1153231015f, +0.2986483765f,
        +0.8948953014f, -0.1705140594f, +0.0822075621f, -0.0453165853f,
        +0.0252021486f, -0.0134457585f, +0.0066649691f, -0.0029803734f,
        +0.0011595035f, -0.0003714116f, +0.0000878918f, -0.0000102134f
    },
    {
        -0.0000008641f, +0.0000326699f, -0.0001782946f, +0.0006361448f,
        -0.0017888932f, +0.0042708158f, -0.0090433922f, +0.0175322749f,
        -0.0320494892f, +0.0573848185f, -0.1083394817f, +0.2775769441f,
        +0.9072455711f, -0.1643526302f, +0.0788931824f, -0.0434551734f,
        +0.0241750876f, -0.0129098044f, +0.0064082111f, -0.0028709744f,
        +0.0011198137f, -0.0003600283f, +0.0000857404f, -0.0000101778f
    },
    {
        -0.0000007094f, +0.0000298696f, -0.0001646973f, +0.0005904165f,
        -0.0016654138f, +0.0039848144f, -0.0084516270f, +0.0164043051f,
        -0.0300078815f, +0.0537246961f, -0.1012382511f, +0.2567128338f,
        +0.9188919652f, -0.1575857157f, +0.0753079865f, -0.0414464831f,
        +0.0230650942f, -0.0123284168f, +0.0061280969f, -0.0027506433f,
        +0.0010756229f, -0.0003470944f, +0.0000831804f, -0.0000100797f
    },
    {
        -0.0000005731f, +0.0000271371f, -0.0001512071f, +0.0005446482f,
        -0.0015410710f, +0.0036954963f, -0.0078509119f, +0.0152563753f,
        -0.0279272044f, +0.0499960908f, -0.0940391131f, +0.2360827804f,
        +0.9298163582f, -0.1502110337f, +0.0714555512f, -0.0392929910f,
        +0.0218735040f, -0.0117021996f, +0.0058248378f, -0.0026194179f,
        +0.0010269141f, -0.0003325890f, +0.0000802001f, -0.0000099150f
    },
    {
        -0.0000004544f, +0.0000244792f, -0.0001378621f, +0.0004989741f,
        -0.0014162353f, +0.0034037249f, -0.0072430303f, +0.0140918639f,
        -0.0258135098f, +0.0462096549f, -0.0867615800f, +0.2157128693f,
        +0.9400017183f, -0.1422273781f, +0.0673401775f, -0.0369975917f,
        +0.0206018939f, -0.0110318900f, +0.0054987142f, -0.0024773684f,
        +0.0009736836f, -0.0003164955f, +0.0000767889f, -0.0000096794f
    },
    {
        -0.0000003526f, +0.0000219020f, -0.0001246981f, +0.0004535241f,
        -0.0012912681f, +0.0031103489f, -0.0066297465f, +0.0129141272f,
        -0.0236728252f, +0.0423759922f, -0.0794249346f, +0.1956284938f,
        +0.9494321403f, -0.1336346333f, +0.0629668935f, -0.0345635983f,
        +0.0192520818f, -0.0103183601f, +0.0051500762f, -0.0023245977f,
        +0.0009159407f, -0.0002988014f, +0.0000729370f, -0.0000093689f
    },
    {
        -0.0000002666f, +0.0000194108f, -0.0001117489f, +0.0004084226f,
        -0.0011665209f, +0.0028162008f, -0.0060128009f, +0.0117264913f,
        -0.0215111387f, +0.0385056324f, -0.0720481930f, +0.1758543135f,
        +0.9580928759f, -0.1244337870f, +0.0583414553f, -0.0319947412f,
        +0.0178261270f, -0.0095626163f, +0.0047793444f, -0.0021612420f,
        +0.0008537086f, -0.0002794990f, +0.0000686360f, -0.0000089795f
    },
    {
        -0.0000001955f, +0.0000170105f, -0.0000990464f, +0.0003637889f,
        -0.0010423343f, +0.0025220943f, -0.0053939061f, +0.0105322435f,
        -0.0193343838f, +0.0346090067f, -0.0646500691f, +0.1564142147f,
        +0.9659703625f, -0.1146269404f, +0.0534703475f, -0.0292951658f,
        +0.0163263284f, -0.0087657999f, +0.0043870101f, -0.0019874714f,
        +0.0007870246f, -0.0002585856f, +0.0000638784f, -0.0000085072f
    },
    {
        -0.0000001380f, +0.0000147054f, -0.0000866200f, +0.0003197369f,
        -0.0009190376f, +0.0022288231f, -0.0047747428f, +0.0093346247f,
        -0.0171484256f, +0.0306964231f, -0.0572489392f, +0.1373312726f,
        +0.9730522488f, -0.1042173167f, +0.0483607803f, -0.0264694294f,
        +0.0147552236f, -0.0079291864f, +0.0039736359f, -0.0018034901f,
        +0.0007159403f, -0.0002360633f, +0.0000586582f, -0.0000079485f
    },
    {
        -0.0000000928f, +0.0000124989f, -0.0000744974f, +0.0002763747f,
        -0.0007969480f, +0.0019371588f, -0.0041569558f, +0.0081368213f,
        -0.0149590466f, +0.0267780441f, -0.0498628084f, +0.1186277152f,
        +0.9793274188f, -0.0932092669f, +0.0430206862f, -0.0235224967f,
        +0.0131155858f, -0.0070541848f, +0.0035398554f, -0.0016095368f,
        +0.0006405222f, -0.0002119393f, +0.0000529707f, -0.0000072998f
    },
    {
        -0.0000000587f, +0.0000103942f, -0.0000627037f, +0.0002338048f,
        -0.0006763702f, +0.0016478494f, -0.0035421504f, +0.0069419578f,
        -0.0127719331f, +0.0228638630f, -0.0425092785f, +0.1003248892f,
        +0.9847860134f, -0.0816082736f, +0.0374587147f, -0.0204597341f,
        +0.0114104213f, -0.0061423367f, +0.0030863730f, -0.0014058851f,
        +0.0005608515f, -0.0001862264f, +0.0000468126f, -0.0000065579f
    },
    {
        -0.0000000340f, +0.0000083935f, -0.0000512623f, +0.0001921235f,
        -0.0005575957f, +0.0013616178f, -0.0029318889f, +0.0057530904f,
        -0.0105926626f, +0.0189636830f, -0.0352055163f, +0.0824432278f,
        +0.9894194491f, -0.0694209524f, +0.0316842250f, -0.0172869036f,
        +0.0096429655f, -0.0051953143f, +0.0026139641f, -0.0011928432f,
        +0.0004770246f, -0.0001589427f, +0.0000401823f, -0.0000057198f
    },
    {
        -0.0000000175f, +0.0000064989f, -0.0000401940f, +0.0001514213f,
        -0.0004409024f, +0.0010791604f, -0.0023276870f, +0.0045731990f,
        -0.0084266909f, +0.0150870956f, -0.0279682243f, +0.0650022208f,
        +0.9932204346f, -0.0566550514f, +0.0257072773f, -0.0140101545f,
        +0.0078166789f, -0.0042149188f, +0.0021234736f, -0.0009707542f,
        +0.0003891534f, -0.0001301121f, +0.0000330795f, -0.0000047827f
    },
    {
        -0.0000000074f, +0.0000047117f, -0.0000295179f, +0.0001117828f,
        -0.0003265541f, +0.0008011455f, -0.0017310111f, +0.0034051820f,
        -0.0062793402f, +0.0112434608f, -0.0208136120f, +0.0480203864f,
        +0.9961829853f, -0.0433194477f, +0.0195386219f, -0.0106360150f,
        +0.0059352416f, -0.0032030780f, +0.0016158161f, -0.0007399959f,
        +0.0002973648f, -0.0000997641f, +0.0000255057f, -0.0000037441f
    },
    {
        -0.0000000022f, +0.0000030327f, -0.0000192507f, +0.0000732860f,
        -0.0002148003f, +0.0005282128f, -0.0011432751f, +0.0022518491f,
        -0.0041557880f, +0.0074418877f, -0.0137573690f, +0.0315152454f,
        +0.9983024341f, -0.0294241423f, +0.0131896874f, -0.0071713821f,
        +0.0040025479f, -0.0021618434f, +0.0010919740f, -0.0005009808f,
        +0.0002018017f, -0.0000679341f, +0.0000174642f, -0.0000026019f
    },
    {
        -0.0000000003f, +0.0000014621f, -0.0000094071f, +0.0000360031f,
        -0.0001058757f, +0.0002609715f, -0.0005658377f, +0.0011159162f,
        -0.0020610560f, +0.0036912163f, -0.0068146398f, +0.0155032976f,
        +0.9995754412f, -0.0149802521f, +0.0066725664f, -0.0036235103f,
        +0.0020226998f, -0.0010933876f, +0.0005529969f, -0.0002541553f,
        +0.0001026222f, -0.0000346636f, +0.0000089600f, -0.0000013543f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    }
};

_Alignas(64) static float const kernels_sinc_32_linear [64][32] =
{
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000005207f, +0.0000029239f, -0.0000097996f, +0.0000261216f,
        -0.0000600592f, +0.0001238576f, -0.0002347374f, +0.0004159416f,
        -0.0006985008f, +0.0011252905f, -0.0017615398f, +0.0027239256f,
        -0.0042717353f, +0.0071745349f, -0.0152564942f, +0.9995798883f,
        +0.0157713998f, -0.0073105332f, +0.0043362153f, -0.0027620098f,
        +0.0017862360f, -0.0011418343f, +0.0007095631f, -0.0004231645f,
        +0.0002392678f, -0.0001265481f, +0.0000615485f, -0.0000268750f,
        +0.0000101387f, -0.0000030541f, +0.0000005594f, -0.0000000001f
    },
    {
        -0.0000010026f, +0.0000057136f, -0.0000192432f, +0.0000514387f,
        -0.0001185038f, +0.0002447554f, -0.0004644190f, +0.0008237112f,
        -0.0013843243f, +0.0022314159f, -0.0034942741f, +0.0054034615f,
        -0.0084693708f, +0.0141983828f, -0.0299840343f, +0.9983202001f,
        +0.0320426187f, -0.0147417944f, +0.0087269885f, -0.0055556148f,
        +0.0035929384f, -0.0022975099f, +0.0014285192f, -0.0008525675f,
        +0.0004825188f, -0.0002555043f, +0.0001244539f, -0.0000544490f,
        +0.0000205978f, -0.0000062340f, +0.0000011575f, -0.0000000009f
    },
    {
        -0.0000014462f, +0.0000083660f, -0.0000283155f, +0.0000759042f,
        -0.0001752162f, +0.0003624375f, -0.0006885427f, +0.0012223968f,
        -0.0020559131f, +0.0033158410f, -0.0051942225f, +0.0080324992f,
        -0.0125836072f, +0.0210574590f, -0.0441696258f, +0.9962228743f,
        +0.0487975515f, -0.0222779562f, +0.0131621155f, -0.0083741567f,
        +0.0054157663f, -0.0034642492f, +0.0021551486f, -0.0012871910f,
        +0.0007291846f, -0.0003865737f, +0.0001885769f, -0.0000826639f,
        +0.0000313572f, -0.0000095346f, +0.0000017939f, -0.0000000031f
    },
    {
        -0.0000018517f, +0.0000108782f, -0.0000370024f, +0.0000994745f,
        -0.0002300860f, +0.0004766622f, -0.0009066304f, +0.0016111255f,
        -0.0027117704f, +0.0043761229f, -0.0068575422f, +0.0106051395f,
        -0.0166054843f, +0.0277383040f, -0.0578013437f, +0.9932911391f,
        +0.0660190952f, -0.0299026598f, +0.0176311308f, -0.0112108219f,
        +0.0072502774f, -0.0046392058f, +0.0028876843f, -0.0017259861f,
        +0.0009786769f, -0.0005194492f, +0.0002537715f, -0.0001114583f,
        +0.0000423952f, -0.0000129504f, +0.0000024683f, -0.0000000074f
    },
    {
        -0.0000022196f, +0.0000132482f, -0.0000452920f, +0.0001221095f,
        -0.0002830104f, +0.0005872016f, -0.0011182288f, +0.0019890651f,
        -0.0033504630f, +0.0054099152f, -0.0084805357f, +0.0131157041f,
        -0.0205263975f, +0.0342281032f, -0.0708683441f, +0.9895295048f,
        +0.0836891719f, -0.0375990419f, +0.0221233294f, -0.0140586566f,
        +0.0090919379f, -0.0058194706f, +0.0036243168f, -0.0021678749f,
        +0.0012303887f, -0.0006538126f, +0.0003198851f, -0.0001407672f,
        +0.0000536885f, -0.0000164749f, +0.0000031800f, -0.0000000143f
    },
    {
        -0.0000025508f, +0.0000154743f, -0.0000531733f, +0.0001437731f,
        -0.0003338946f, +0.0006938430f, -0.0013229101f, +0.0023554255f,
        -0.0039706245f, +0.0064149731f, -0.0100596584f, +0.0155587475f,
        -0.0243381137f, +0.0405147043f, -0.0833608751f, +0.9849437565f,
        +0.1017887543f, -0.0453497634f, +0.0266277894f, -0.0169105828f,
        +0.0109361328f, -0.0070020787f, +0.0043631974f, -0.0026117530f,
        +0.0014836958f, -0.0007893347f, +0.0003867591f, -0.0001705226f,
        +0.0000652124f, -0.0000201014f, +0.0000039282f, -0.0000000247f
    },
    {
        -0.0000028458f, +0.0000175554f, -0.0000606374f, +0.0001644328f,
        -0.0003826515f, +0.0007963882f, -0.0015202727f, +0.0027094601f,
        -0.0045709571f, +0.0073891574f, -0.0115915252f, +0.0179290671f,
        -0.0280327858f, +0.0465866345f, -0.0952702844f, +0.9795409427f,
        +0.1202978923f, -0.0531370391f, +0.0311333953f, -0.0197594142f,
        +0.0127781771f, -0.0081840163f, +0.0051024430f, -0.0030564921f,
        +0.0017379579f, -0.0009256763f, +0.0004542286f, -0.0002006532f,
        +0.0000769406f, -0.0000238225f, +0.0000047118f, -0.0000000390f
    },
    {
        -0.0000031057f, +0.0000194911f, -0.0000676766f, +0.0001840599f,
        -0.0004292017f, +0.0008946542f, -0.0017099421f, +0.0030504673f,
        -0.0051502343f, +0.0083304389f, -0.0130729173f, +0.0202217128f,
        -0.0316029663f, +0.0524331143f, -0.1065890254f, +0.9733293620f,
        +0.1391957421f, -0.0609426692f, +0.0356288621f, -0.0225978729f,
        +0.0146133261f, -0.0093622278f, +0.0058401401f, -0.0035009423f,
        +0.0019925201f, -0.0010624889f, +0.0005221236f, -0.0002310848f,
        +0.0000888454f, -0.0000276299f, +0.0000055297f, -0.0000000580f
    },
    {
        -0.0000033315f, +0.0000212811f, -0.0000742848f, +0.0002026293f,
        -0.0004734741f, +0.0009884734f, -0.0018915707f, +0.0033777909f,
        -0.0057073032f, +0.0092369015f, -0.0145007878f, +0.0224319962f,
        -0.0350416201f, +0.0580440707f, -0.1173106610f, +0.9663185485f,
        +0.1584605976f, -0.0687480713f, +0.0401027606f, -0.0254186063f,
        +0.0164367876f, -0.0105336229f, +0.0065743499f, -0.0039439351f,
        +0.0022467144f, -0.0011994152f, +0.0005902684f, -0.0002617400f,
        +0.0001008975f, -0.0000315150f, +0.0000063804f, -0.0000000821f
    },
    {
        -0.0000035242f, +0.0000229260f, -0.0000804573f, +0.0002201196f,
        -0.0005154051f, +0.0010776936f, -0.0020648391f, +0.0036908219f,
        -0.0062410859f, +0.0101067461f, -0.0158722675f, +0.0245554989f,
        -0.0383421352f, +0.0634101488f, -0.1274298647f, +0.9585192530f,
        +0.1780699229f, -0.0765343146f, +0.0445435428f, -0.0282142049f,
        +0.0182437329f, -0.0116950847f, +0.0073031125f, -0.0043842862f,
        +0.0024998613f, -0.0013360904f, +0.0006584829f, -0.0002925390f,
        +0.0001130665f, -0.0000354687f, +0.0000072623f, -0.0000001119f
    },
    {
        -0.0000036851f, +0.0000244266f, -0.0000861911f, +0.0002365129f,
        -0.0005549393f, +0.0011621779f, -0.0022294555f, +0.0039889987f,
        -0.0067505814f, +0.0109382931f, -0.0171846692f, +0.0265880796f,
        -0.0414983334f, +0.0685227209f, -0.1369424201f, +0.9499434239f,
        +0.1980003865f, -0.0842821538f, +0.0489395680f, -0.0309772199f,
        +0.0200293089f, -0.0128434771f, +0.0080244521f, -0.0048207981f,
        +0.0027512715f, -0.0014721425f, +0.0007265821f, -0.0003233993f,
        +0.0001253205f, -0.0000394810f, +0.0000081735f, -0.0000001478f
    },
    {
        -0.0000038155f, +0.0000257842f, -0.0000914842f, +0.0002517952f,
        -0.0005920292f, +0.0012418054f, -0.0023851568f, +0.0042718082f,
        -0.0072348669f, +0.0117299851f, -0.0184354926f, +0.0285258809f,
        -0.0445044788f, +0.0733738949f, -0.1458452182f, +0.9406041843f,
        +0.2182278982f, -0.0919720658f, +0.0532791299f, -0.0337001814f,
        +0.0217886505f, -0.0139756523f, +0.0087363817f, -0.0052522633f,
        +0.0030002475f, -0.0016071935f, +0.0007943773f, -0.0003542359f,
        +0.0001376263f, -0.0000435414f, +0.0000091120f, -0.0000001904f
    },
    {
        -0.0000039168f, +0.0000270005f, -0.0000963362f, +0.0002659558f,
        -0.0006266348f, +0.0013164704f, -0.0025317081f, +0.0045387857f,
        -0.0076930993f, +0.0124803894f, -0.0196224279f, +0.0303653355f,
        -0.0473552859f, +0.0779565201f, -0.1541362523f, +0.9305158081f,
        +0.2387276471f, -0.0995842856f, +0.0575504837f, -0.0363756173f,
        +0.0235168924f, -0.0150884595f, +0.0094369081f, -0.0056774674f,
        +0.0032460851f, -0.0017408604f, +0.0008616760f, -0.0003849616f,
        +0.0001499496f, -0.0000476391f, +0.0000100754f, -0.0000002399f
    },
    {
        -0.0000039903f, +0.0000280778f, -0.0001007481f, +0.0002789874f,
        -0.0006587244f, +0.0013860828f, -0.0026689034f, +0.0047895162f,
        -0.0081245157f, +0.0131881995f, -0.0207433590f, +0.0321031706f,
        -0.0500459259f, +0.0822641919f, -0.1618146106f, +0.9196936933f,
        +0.2594741405f, -0.1070988448f, +0.0617418740f, -0.0389960713f,
        +0.0252091818f, -0.0161787528f, +0.0101240376f, -0.0060951921f,
        +0.0034880756f, -0.0018727558f, +0.0009282825f, -0.0004154871f,
        +0.0001622550f, -0.0000517625f, +0.0000110614f, -0.0000002967f
    },
    {
        -0.0000040376f, +0.0000290184f, -0.0001047220f, +0.0002908862f,
        -0.0006882734f, +0.0014505681f, -0.0027965649f, +0.0050236339f,
        -0.0085284345f, +0.0138522371f, -0.0217963663f, +0.0337364127f,
        -0.0525720324f, +0.0862912547f, -0.1688804675f, +0.9081543333f,
        +0.2804412457f, -0.1144956095f, +0.0658415632f, -0.0415541227f,
        +0.0268606912f, -0.0172433990f, +0.0107957804f, -0.0065042183f,
        +0.0037255074f, -0.0020024891f, +0.0009939987f, -0.0004457215f,
        +0.0001745058f, -0.0000558996f, +0.0000120672f, -0.0000003610f
    },
    {
        -0.0000040602f, +0.0000298253f, -0.0001082614f, +0.0003016517f,
        -0.0007152650f, +0.0015098671f, -0.0029145436f, +0.0052408225f,
        -0.0089042557f, +0.0144714529f, -0.0227797289f, +0.0352623908f,
        -0.0549297053f, +0.0900328028f, -0.1753350721f, +0.8959152867f,
        +0.3016022321f, -0.1217543201f, +0.0698378594f, -0.0440424048f,
        +0.0284666307f, -0.0182792868f, +0.0114501565f, -0.0069033295f,
        +0.0039576680f, -0.0021296677f, +0.0010586240f, -0.0004755720f,
        +0.0001866646f, -0.0000600378f, +0.0000130899f, -0.0000004331f
    },
    {
        -0.0000040596f, +0.0000305016f, -0.0001113708f, +0.0003112867f,
        -0.0007396899f, +0.0015639355f, -0.0030227189f, +0.0054408155f,
        -0.0092514613f, +0.0150449276f, -0.0236919265f, +0.0366787390f,
        -0.0571155141f, +0.0934846805f, -0.1811807348f, +0.8829951445f,
        +0.3229298156f, -0.1288546308f, +0.0737191460f, -0.0464536244f,
        +0.0300222612f, -0.0192833342f, +0.0120852010f, -0.0072913150f,
        +0.0041838456f, -0.0022538975f, +0.0011219564f, -0.0005049446f,
        +0.0001986928f, -0.0000641641f, +0.0000141264f, -0.0000005130f
    },
    {
        -0.0000040374f, +0.0000310508f, -0.0001140559f, +0.0003197968f,
        -0.0007615459f, +0.0016127444f, -0.0031209983f, +0.0056233957f,
        -0.0095696152f, +0.0155718723f, -0.0245316401f, +0.0379833987f,
        -0.0591264998f, +0.0966434792f, -0.1864208121f, +0.8694134958f,
        +0.3443962032f, -0.1357761508f, +0.0774739101f, -0.0487805812f,
        +0.0315229069f, -0.0202524976f, +0.0126989692f, -0.0076669734f,
        +0.0044033313f, -0.0023747843f, +0.0011837925f, -0.0005337440f,
        +0.0002105512f, -0.0000682650f, +0.0000151734f, -0.0000006009f
    },
    {
        -0.0000039953f, +0.0000314768f, -0.0001163236f, +0.0003271911f,
        -0.0007808379f, +0.0016562792f, -0.0032093171f, +0.0057883951f,
        -0.0098583634f, +0.0160516289f, -0.0252977529f, +0.0391746190f,
        -0.0609601755f, +0.0995065343f, -0.1910596896f, +0.8551908916f,
        +0.3659731397f, -0.1424984852f, +0.0810907720f, -0.0510161870f,
        +0.0329639683f, -0.0211837802f, +0.0132895422f, -0.0080291153f,
        +0.0046154211f, -0.0024919352f, +0.0012439284f, -0.0005618741f,
        +0.0002221999f, -0.0000723266f, +0.0000162272f, -0.0000006967f
    },
    {
        -0.0000039349f, +0.0000317835f, -0.0001181816f, +0.0003334812f,
        -0.0007975778f, +0.0016945401f, -0.0032876382f, +0.0059356945f,
        -0.0101174334f, +0.0164836697f, -0.0259893504f, +0.0402509577f,
        -0.0626145258f, +0.1020719195f, -0.1951027626f, +0.8403488071f,
        +0.3876319556f, -0.1490012763f, +0.0845585143f, -0.0531534849f,
        +0.0343409350f, -0.0220742398f, +0.0138550322f, -0.0083765676f,
        +0.0048194177f, -0.0026049590f, +0.0013021603f, -0.0005892380f,
        +0.0002335981f, -0.0000763346f, +0.0000172844f, -0.0000008004f
    },
    {
        -0.0000038577f, +0.0000319753f, -0.0001196387f, +0.0003386816f,
        -0.0008117841f, +0.0017275411f, -0.0033559517f, +0.0060652228f,
        -0.0103466336f, +0.0168675968f, -0.0266057197f, +0.0412112804f,
        -0.0640880055f, +0.1043384402f, -0.1985564155f, +0.8249096016f,
        +0.4093436148f, -0.1552642462f, +0.0878661114f, -0.0551856689f,
        +0.0356493981f, -0.0229209982f, +0.0143935883f, -0.0087081763f,
        +0.0050146327f, -0.0027134677f, +0.0013582849f, -0.0006157385f,
        +0.0002447048f, -0.0000802743f, +0.0000183408f, -0.0000009116f
    },
    {
        -0.0000037654f, +0.0000320566f, -0.0001207046f, +0.0003428097f,
        -0.0008234818f, +0.0017553103f, -0.0034142742f, +0.0061769565f,
        -0.0105458525f, +0.0172031415f, -0.0271463490f, +0.0420547593f,
        -0.0653795370f, +0.1063056249f, -0.2014279988f, +0.8088964781f,
        +0.4310787648f, -0.1612672387f, +0.0910027583f, -0.0571061022f,
        +0.0368850636f, -0.0237212487f, +0.0149034014f, -0.0090228097f,
        +0.0052003883f, -0.0028170776f, +0.0014120998f, -0.0006412784f,
        +0.0002554785f, -0.0000841308f, +0.0000193925f, -0.0000010303f
    },
    {
        -0.0000036596f, +0.0000320322f, -0.0001213899f, +0.0003458850f,
        -0.0008327023f, +0.0017778891f, -0.0034626485f, +0.0062709193f,
        -0.0107150582f, +0.0174901629f, -0.0276109257f, +0.0427808712f,
        -0.0664885062f, +0.1079737154f, -0.2037258049f, +0.7923334400f,
        +0.4528077868f, -0.1669902617f, +0.0939579000f, -0.0589083373f,
        +0.0380437643f, -0.0244722649f, +0.0153827098f, -0.0093193626f,
        +0.0053760197f, -0.0029154106f, +0.0014634045f, -0.0006657604f,
        +0.0002658773f, -0.0000878887f, +0.0000204351f, -0.0000011560f
    },
    {
        -0.0000035419f, +0.0000319068f, -0.0001217059f, +0.0003479300f,
        -0.0008394829f, +0.0017953320f, -0.0035011426f, +0.0063471803f,
        -0.0108542964f, +0.0177286465f, -0.0279993348f, +0.0433893949f,
        -0.0674147577f, +0.1093436553f, -0.2054590417f, +0.7752452475f,
        +0.4745008464f, -0.1724135299f, +0.0967212602f, -0.0605861334f,
        +0.0391214722f, -0.0251714086f, +0.0158298048f, -0.0095967586f,
        +0.0055408769f, -0.0030080950f, +0.0015120008f, -0.0006890878f,
        +0.0002758591f, -0.0000915326f, +0.0000214642f, -0.0000012884f
    },
    {
        -0.0000034138f, +0.0000316855f, -0.0001216646f, +0.0003489689f,
        -0.0008438668f, +0.0018077060f, -0.0035298496f, +0.0064058541f,
        -0.0109636899f, +0.0179187023f, -0.0283116561f, +0.0438804072f,
        -0.0681585892f, +0.1104170775f, -0.2066378055f, +0.7576573720f,
        +0.4961279457f, -0.1775175075f, +0.0992828702f, -0.0621334761f,
        +0.0401143112f, -0.0258161382f, +0.0162430356f, -0.0098539544f,
        +0.0056943267f, -0.0030947668f, +0.0015576934f, -0.0007111646f,
        +0.0002853819f, -0.0000950467f, +0.0000224753f, -0.0000014269f
    },
    {
        -0.0000032769f, +0.0000313733f, -0.0001212787f, +0.0003490285f,
        -0.0008459024f, +0.0018150904f, -0.0035488864f, +0.0064470989f,
        -0.0110434366f, +0.0180605630f, -0.0285481619f, +0.0442542795f,
        -0.0687207444f, +0.1111962899f, -0.2072730511f, +0.7395959498f,
        +0.5176589755f, -0.1822829501f, +0.1016330968f, -0.0635445944f,
        +0.0410185683f, -0.0264040162f, +0.0166208146f, -0.0100899423f,
        +0.0058357550f, -0.0031750710f, +0.0016002906f, -0.0007318958f,
        +0.0002944037f, -0.0000984150f, +0.0000234636f, -0.0000015710f
    },
    {
        -0.0000031326f, +0.0000309756f, -0.0001205616f, +0.0003481375f,
        -0.0008456438f, +0.0018175760f, -0.0035583934f, +0.0064711159f,
        -0.0110938078f, +0.0181545812f, -0.0287093131f, +0.0445116725f,
        -0.0691024048f, +0.1116842599f, -0.2073765618f, +0.7210877343f,
        +0.5390637676f, -0.1866909480f, +0.1037626706f, -0.0648139793f,
        +0.0418307061f, -0.0269327175f, +0.0169616226f, -0.0103037542f,
        +0.0059645683f, -0.0032486624f, +0.0016396046f, -0.0007511877f,
        +0.0003028827f, -0.0001016216f, +0.0000244242f, -0.0000017200f
    },
    {
        -0.0000029823f, +0.0000304975f, -0.0001195271f, +0.0003463262f,
        -0.0008431497f, +0.0018152648f, -0.0035585334f, +0.0064781475f,
        -0.0111151466f, +0.0182012270f, -0.0287957559f, +0.0446535305f,
        -0.0693051816f, +0.1118845978f, -0.2069609168f, +0.7021600472f,
        +0.5603121480f, -0.1907229674f, +0.1056627132f, -0.0659364007f,
        +0.0425473736f, -0.0274000365f, +0.0172640137f, -0.0104944643f,
        +0.0060801961f, -0.0033152073f, +0.0016754525f, -0.0007689484f,
        +0.0003107774f, -0.0001046503f, +0.0000253522f, -0.0000018732f
    },
    {
        -0.0000028275f, +0.0000299444f, -0.0001181895f, +0.0003436271f,
        -0.0008384836f, +0.0018082698f, -0.0035494907f, +0.0064684763f,
        -0.0111078656f, +0.0182010852f, -0.0288083171f, +0.0446810756f,
        -0.0693311049f, +0.1118015388f, -0.2060394583f, +0.6828407290f,
        +0.5813739902f, -0.1943608923f, +0.1073247642f, -0.0669069243f,
        +0.0431654175f, -0.0278038950f, +0.0175266204f, -0.0106611923f,
        +0.0061820926f, -0.0033743840f, +0.0017076571f, -0.0007850876f,
        +0.0003180467f, -0.0001074850f, +0.0000262425f, -0.0000020296f
    },
    {
        -0.0000026695f, +0.0000293217f, -0.0001165638f, +0.0003400740f,
        -0.0008317135f, +0.0017967135f, -0.0035314703f, +0.0064424238f,
        -0.0110724448f, +0.0181548517f, -0.0287480000f, +0.0445958010f,
        -0.0691826138f, +0.1114399238f, -0.2046262569f, +0.6631580886f,
        +0.6022192683f, -0.1975870664f, +0.1087408073f, -0.0677209282f,
        +0.0436818930f, -0.0281423488f, +0.0177481581f, -0.0108031066f,
        +0.0062697388f, -0.0034258843f, +0.0017360466f, -0.0007995176f,
        +0.0003246503f, -0.0001101097f, +0.0000270899f, -0.0000021885f
    },
    {
        -0.0000025096f, +0.0000286348f, -0.0001146649f, +0.0003357023f,
        -0.0008229114f, +0.0017807286f, -0.0035046972f, +0.0064003484f,
        -0.0110094295f, +0.0180633305f, -0.0286159792f, +0.0443994630f,
        -0.0688625447f, +0.1108051793f, -0.2027360754f, +0.6431408520f,
        +0.6228181105f, -0.2003843330f, +0.1099032962f, -0.0683741186f,
        +0.0440940739f, -0.0284135952f, +0.0179274298f, -0.0109194275f,
        +0.0063426441f, -0.0034694147f, +0.0017604563f, -0.0008121530f,
        +0.0003305484f, -0.0001125083f, +0.0000278891f, -0.0000023489f
    },
    {
        -0.0000023489f, +0.0000278891f, -0.0001125083f, +0.0003305484f,
        -0.0008121530f, +0.0017604563f, -0.0034694147f, +0.0063426441f,
        -0.0109194275f, +0.0179274298f, -0.0284135952f, +0.0440940739f,
        -0.0683741186f, +0.1099032962f, -0.2003843330f, +0.6228181105f,
        +0.6431408520f, -0.2027360754f, +0.1108051793f, -0.0688625447f,
        +0.0443994630f, -0.0286159792f, +0.0180633305f, -0.0110094295f,
        +0.0064003484f, -0.0035046972f, +0.0017807286f, -0.0008229114f,
        +0.0003357023f, -0.0001146649f, +0.0000286348f, -0.0000025096f
    },
    {
        -0.0000021885f, +0.0000270899f, -0.0001101097f, +0.0003246503f,
        -0.0007995176f, +0.0017360466f, -0.0034258843f, +0.0062697388f,
        -0.0108031066f, +0.0177481581f, -0.0281423488f, +0.0436818930f,
        -0.0677209282f, +0.1087408073f, -0.1975870664f, +0.6022192683f,
        +0.6631580886f, -0.2046262569f, +0.1114399238f, -0.0691826138f,
        +0.0445958010f, -0.0287480000f, +0.0181548517f, -0.0110724448f,
        +0.0064424238f, -0.0035314703f, +0.0017967135f, -0.0008317135f,
        +0.0003400740f, -0.0001165638f, +0.0000293217f, -0.0000026695f
    },
    {
        -0.0000020296f, +0.0000262425f, -0.0001074850f, +0.0003180467f,
        -0.0007850876f, +0.0017076571f, -0.0033743840f, +0.0061820926f,
        -0.0106611923f, +0.0175266204f, -0.0278038950f, +0.0431654175f,
        -0.0669069243f, +0.1073247642f, -0.1943608923f, +0.5813739902f,
        +0.6828407290f, -0.2060394583f, +0.1118015388f, -0.0693311049f,
        +0.0446810756f, -0.0288083171f, +0.0182010852f, -0.0111078656f,
        +0.0064684763f, -0.0035494907f, +0.0018082698f, -0.0008384836f,
        +0.0003436271f, -0.0001181895f, +0.0000299444f, -0.0000028275f
    },
    {
        -0.0000018732f, +0.0000253522f, -0.0001046503f, +0.0003107774f,
        -0.0007689484f, +0.0016754525f, -0.0033152073f, +0.0060801961f,
        -0.0104944643f, +0.0172640137f, -0.0274000365f, +0.0425473736f,
        -0.0659364007f, +0.1056627132f, -0.1907229674f, +0.5603121480f,
        +0.7021600472f, -0.2069609168f, +0.1118845978f, -0.0693051816f,
        +0.0446535305f, -0.0287957559f, +0.0182012270f, -0.0111151466f,
        +0.0064781475f, -0.0035585334f, +0.0018152648f, -0.0008431497f,
        +0.0003463262f, -0.0001195271f, +0.0000304975f, -0.0000029823f
    },
    {
        -0.0000017200f, +0.0000244242f, -0.0001016216f, +0.0003028827f,
        -0.0007511877f, +0.0016396046f, -0.0032486624f, +0.0059645683f,
        -0.0103037542f, +0.0169616226f, -0.0269327175f, +0.0418307061f,
        -0.0648139793f, +0.1037626706f, -0.1866909480f, +0.5390637676f,
        +0.7210877343f, -0.2073765618f, +0.1116842599f, -0.0691024048f,
        +0.0445116725f, -0.0287093131f, +0.0181545812f, -0.0110938078f,
        +0.0064711159f, -0.0035583934f, +0.0018175760f, -0.0008456438f,
        +0.0003481375f, -0.0001205616f, +0.0000309756f, -0.0000031326f
    },
    {
        -0.0000015710f, +0.0000234636f, -0.0000984150f, +0.0002944037f,
        -0.0007318958f, +0.0016002906f, -0.0031750710f, +0.0058357550f,
        -0.0100899423f, +0.0166208146f, -0.0264040162f, +0.0410185683f,
        -0.0635445944f, +0.1016330968f, -0.1822829501f, +0.5176589755f,
        +0.7395959498f, -0.2072730511f, +0.1111962899f, -0.0687207444f,
        +0.0442542795f, -0.0285481619f, +0.0180605630f, -0.0110434366f,
        +0.0064470989f, -0.0035488864f, +0.0018150904f, -0.0008459024f,
        +0.0003490285f, -0.0001212787f, +0.0000313733f, -0.0000032769f
    },
    {
        -0.0000014269f, +0.0000224753f, -0.0000950467f, +0.0002853819f,
        -0.0007111646f, +0.0015576934f, -0.0030947668f, +0.0056943267f,
        -0.0098539544f, +0.0162430356f, -0.0258161382f, +0.0401143112f,
        -0.0621334761f, +0.0992828702f, -0.1775175075f, +0.4961279457f,
        +0.7576573720f, -0.2066378055f, +0.1104170775f, -0.0681585892f,
        +0.0438804072f, -0.0283116561f, +0.0179187023f, -0.0109636899f,
        +0.0064058541f, -0.0035298496f, +0.0018077060f, -0.0008438668f,
        +0.0003489689f, -0.0001216646f, +0.0000316855f, -0.0000034138f
    },
    {
        -0.0000012884f, +0.0000214642f, -0.0000915326f, +0.0002758591f,
        -0.0006890878f, +0.0015120008f, -0.0030080950f, +0.0055408769f,
        -0.0095967586f, +0.0158298048f, -0.0251714086f, +0.0391214722f,
        -0.0605861334f, +0.0967212602f, -0.1724135299f, +0.4745008464f,
        +0.7752452475f, -0.2054590417f, +0.1093436553f, -0.0674147577f,
        +0.0433893949f, -0.0279993348f, +0.0177286465f, -0.0108542964f,
        +0.0063471803f, -0.0035011426f, +0.0017953320f, -0.0008394829f,
        +0.0003479300f, -0.0001217059f, +0.0000319068f, -0.0000035419f
    },
    {
        -0.0000011560f, +0.0000204351f, -0.0000878887f, +0.0002658773f,
        -0.0006657604f, +0.0014634045f, -0.0029154106f, +0.0053760197f,
        -0.0093193626f, +0.0153827098f, -0.0244722649f, +0.0380437643f,
        -0.0589083373f, +0.0939579000f, -0.1669902617f, +0.4528077868f,
        +0.7923334400f, -0.2037258049f, +0.1079737154f, -0.0664885062f,
        +0.0427808712f, -0.0276109257f, +0.0174901629f, -0.0107150582f,
        +0.0062709193f, -0.0034626485f, +0.0017778891f, -0.0008327023f,
        +0.0003458850f, -0.0001213899f, +0.0000320322f, -0.0000036596f
    },
    {
        -0.0000010303f, +0.0000193925f, -0.0000841308f, +0.0002554785f,
        -0.0006412784f, +0.0014120998f, -0.0028170776f, +0.0052003883f,
        -0.0090228097f, +0.0149034014f, -0.0237212487f, +0.0368850636f,
        -0.0571061022f, +0.0910027583f, -0.1612672387f, +0.4310787648f,
        +0.8088964781f, -0.2014279988f, +0.1063056249f, -0.0653795370f,
        +0.0420547593f, -0.0271463490f, +0.0172031415f, -0.0105458525f,
        +0.0061769565f, -0.0034142742f, +0.0017553103f, -0.0008234818f,
        +0.0003428097f, -0.0001207046f, +0.0000320566f, -0.0000037654f
    },
    {
        -0.0000009116f, +0.0000183408f, -0.0000802743f, +0.0002447048f,
        -0.0006157385f, +0.0013582849f, -0.0027134677f, +0.0050146327f,
        -0.0087081763f, +0.0143935883f, -0.0229209982f, +0.0356493981f,
        -0.0551856689f, +0.0878661114f, -0.1552642462f, +0.4093436148f,
        +0.8249096016f, -0.1985564155f, +0.1043384402f, -0.0640880055f,
        +0.0412112804f, -0.0266057197f, +0.0168675968f, -0.0103466336f,
        +0.0060652228f, -0.0033559517f, +0.0017275411f, -0.0008117841f,
        +0.0003386816f, -0.0001196387f, +0.0000319753f, -0.0000038577f
    },
    {
        -0.0000008004f, +0.0000172844f, -0.0000763346f, +0.0002335981f,
        -0.0005892380f, +0.0013021603f, -0.0026049590f, +0.0048194177f,
        -0.0083765676f, +0.0138550322f, -0.0220742398f, +0.0343409350f,
        -0.0531534849f, +0.0845585143f, -0.1490012763f, +0.3876319556f,
        +0.8403488071f, -0.1951027626f, +0.1020719195f, -0.0626145258f,
        +0.0402509577f, -0.0259893504f, +0.0164836697f, -0.0101174334f,
        +0.0059356945f, -0.0032876382f, +0.0016945401f, -0.0007975778f,
        +0.0003334812f, -0.0001181816f, +0.0000317835f, -0.0000039349f
    },
    {
        -0.0000006967f, +0.0000162272f, -0.0000723266f, +0.0002221999f,
        -0.0005618741f, +0.0012439284f, -0.0024919352f, +0.0046154211f,
        -0.0080291153f, +0.0132895422f, -0.0211837802f, +0.0329639683f,
        -0.0510161870f, +0.0810907720f, -0.1424984852f, +0.3659731397f,
        +0.8551908916f, -0.1910596896f, +0.0995065343f, -0.0609601755f,
        +0.0391746190f, -0.0252977529f, +0.0160516289f, -0.0098583634f,
        +0.0057883951f, -0.0032093171f, +0.0016562792f, -0.0007808379f,
        +0.0003271911f, -0.0001163236f, +0.0000314768f, -0.0000039953f
    },
    {
        -0.0000006009f, +0.0000151734f, -0.0000682650f, +0.0002105512f,
        -0.0005337440f, +0.0011837925f, -0.0023747843f, +0.0044033313f,
        -0.0076669734f, +0.0126989692f, -0.0202524976f, +0.0315229069f,
        -0.0487805812f, +0.0774739101f, -0.1357761508f, +0.3443962032f,
        +0.8694134958f, -0.1864208121f, +0.0966434792f, -0.0591264998f,
        +0.0379833987f, -0.0245316401f, +0.0155718723f, -0.0095696152f,
        +0.0056233957f, -0.0031209983f, +0.0016127444f, -0.0007615459f,
        +0.0003197968f, -0.0001140559f, +0.0000310508f, -0.0000040374f
    },
    {
        -0.0000005130f, +0.0000141264f, -0.0000641641f, +0.0001986928f,
        -0.0005049446f, +0.0011219564f, -0.0022538975f, +0.0041838456f,
        -0.0072913150f, +0.0120852010f, -0.0192833342f, +0.0300222612f,
        -0.0464536244f, +0.0737191460f, -0.1288546308f, +0.3229298156f,
        +0.8829951445f, -0.1811807348f, +0.0934846805f, -0.0571155141f,
        +0.0366787390f, -0.0236919265f, +0.0150449276f, -0.0092514613f,
        +0.0054408155f, -0.0030227189f, +0.0015639355f, -0.0007396899f,
        +0.0003112867f, -0.0001113708f, +0.0000305016f, -0.0000040596f
    },
    {
        -0.0000004331f, +0.0000130899f, -0.0000600378f, +0.0001866646f,
        -0.0004755720f, +0.0010586240f, -0.0021296677f, +0.0039576680f,
        -0.0069033295f, +0.0114501565f, -0.0182792868f, +0.0284666307f,
        -0.0440424048f, +0.0698378594f, -0.1217543201f, +0.3016022321f,
        +0.8959152867f, -0.1753350721f, +0.0900328028f, -0.0549297053f,
        +0.0352623908f, -0.0227797289f, +0.0144714529f, -0.0089042557f,
        +0.0052408225f, -0.0029145436f, +0.0015098671f, -0.0007152650f,
        +0.0003016517f, -0.0001082614f, +0.0000298253f, -0.0000040602f
    },
    {
        -0.0000003610f, +0.0000120672f, -0.0000558996f, +0.0001745058f,
        -0.0004457215f, +0.0009939987f, -0.0020024891f, +0.0037255074f,
        -0.0065042183f, +0.0107957804f, -0.0172433990f, +0.0268606912f,
        -0.0415541227f, +0.0658415632f, -0.1144956095f, +0.2804412457f,
        +0.9081543333f, -0.1688804675f, +0.0862912547f, -0.0525720324f,
        +0.0337364127f, -0.0217963663f, +0.0138522371f, -0.0085284345f,
        +0.0050236339f, -0.0027965649f, +0.0014505681f, -0.0006882734f,
        +0.0002908862f, -0.0001047220f, +0.0000290184f, -0.0000040376f
    },
    {
        -0.0000002967f, +0.0000110614f, -0.0000517625f, +0.0001622550f,
        -0.0004154871f, +0.0009282825f, -0.0018727558f, +0.0034880756f,
        -0.0060951921f, +0.0101240376f, -0.0161787528f, +0.0252091818f,
        -0.0389960713f, +0.0617418740f, -0.1070988448f, +0.2594741405f,
        +0.9196936933f, -0.1618146106f, +0.0822641919f, -0.0500459259f,
        +0.0321031706f, -0.0207433590f, +0.0131881995f, -0.0081245157f,
        +0.0047895162f, -0.0026689034f, +0.0013860828f, -0.0006587244f,
        +0.0002789874f, -0.0001007481f, +0.0000280778f, -0.0000039903f
    },
    {
        -0.0000002399f, +0.0000100754f, -0.0000476391f, +0.0001499496f,
        -0.0003849616f, +0.0008616760f, -0.0017408604f, +0.0032460851f,
        -0.0056774674f, +0.0094369081f, -0.0150884595f, +0.0235168924f,
        -0.0363756173f, +0.0575504837f, -0.0995842856f, +0.2387276471f,
        +0.9305158081f, -0.1541362523f, +0.0779565201f, -0.0473552859f,
        +0.0303653355f, -0.0196224279f, +0.0124803894f, -0.0076930993f,
        +0.0045387857f, -0.0025317081f, +0.0013164704f, -0.0006266348f,
        +0.0002659558f, -0.0000963362f, +0.0000270005f, -0.0000039168f
    },
    {
        -0.0000001904f, +0.0000091120f, -0.0000435414f, +0.0001376263f,
        -0.0003542359f, +0.0007943773f, -0.0016071935f, +0.0030002475f,
        -0.0052522633f, +0.0087363817f, -0.0139756523f, +0.0217886505f,
        -0.0337001814f, +0.0532791299f, -0.0919720658f, +0.2182278982f,
        +0.9406041843f, -0.1458452182f, +0.0733738949f, -0.0445044788f,
        +0.0285258809f, -0.0184354926f, +0.0117299851f, -0.0072348669f,
        +0.0042718082f, -0.0023851568f, +0.0012418054f, -0.0005920292f,
        +0.0002517952f, -0.0000914842f, +0.0000257842f, -0.0000038155f
    },
    {
        -0.0000001478f, +0.0000081735f, -0.0000394810f, +0.0001253205f,
        -0.0003233993f, +0.0007265821f, -0.0014721425f, +0.0027512715f,
        -0.0048207981f, +0.0080244521f, -0.0128434771f, +0.0200293089f,
        -0.0309772199f, +0.0489395680f, -0.0842821538f, +0.1980003865f,
        +0.9499434239f, -0.1369424201f, +0.0685227209f, -0.0414983334f,
        +0.0265880796f, -0.0171846692f, +0.0109382931f, -0.0067505814f,
        +0.0039889987f, -0.0022294555f, +0.0011621779f, -0.0005549393f,
        +0.0002365129f, -0.0000861911f, +0.0000244266f, -0.0000036851f
    },
    {
        -0.0000001119f, +0.0000072623f, -0.0000354687f, +0.0001130665f,
        -0.0002925390f, +0.0006584829f, -0.0013360904f, +0.0024998613f,
        -0.0043842862f, +0.0073031125f, -0.0116950847f, +0.0182437329f,
        -0.0282142049f, +0.0445435428f, -0.0765343146f, +0.1780699229f,
        +0.9585192530f, -0.1274298647f, +0.0634101488f, -0.0383421352f,
        +0.0245554989f, -0.0158722675f, +0.0101067461f, -0.0062410859f,
        +0.0036908219f, -0.0020648391f, +0.0010776936f, -0.0005154051f,
        +0.0002201196f, -0.0000804573f, +0.0000229260f, -0.0000035242f
    },
    {
        -0.0000000821f, +0.0000063804f, -0.0000315150f, +0.0001008975f,
        -0.0002617400f, +0.0005902684f, -0.0011994152f, +0.0022467144f,
        -0.0039439351f, +0.0065743499f, -0.0105336229f, +0.0164367876f,
        -0.0254186063f, +0.0401027606f, -0.0687480713f, +0.1584605976f,
        +0.9663185485f, -0.1173106610f, +0.0580440707f, -0.0350416201f,
        +0.0224319962f, -0.0145007878f, +0.0092369015f, -0.0057073032f,
        +0.0033777909f, -0.0018915707f, +0.0009884734f, -0.0004734741f,
        +0.0002026293f, -0.0000742848f, +0.0000212811f, -0.0000033315f
    },
    {
        -0.0000000580f, +0.0000055297f, -0.0000276299f, +0.0000888454f,
        -0.0002310848f, +0.0005221236f, -0.0010624889f, +0.0019925201f,
        -0.0035009423f, +0.0058401401f, -0.0093622278f, +0.0146133261f,
        -0.0225978729f, +0.0356288621f, -0.0609426692f, +0.1391957421f,
        +0.9733293620f, -0.1065890254f, +0.0524331143f, -0.0316029663f,
        +0.0202217128f, -0.0130729173f, +0.0083304389f, -0.0051502343f,
        +0.0030504673f, -0.0017099421f, +0.0008946542f, -0.0004292017f,
        +0.0001840599f, -0.0000676766f, +0.0000194911f, -0.0000031057f
    },
    {
        -0.0000000390f, +0.0000047118f, -0.0000238225f, +0.0000769406f,
        -0.0002006532f, +0.0004542286f, -0.0009256763f, +0.0017379579f,
        -0.0030564921f, +0.0051024430f, -0.0081840163f, +0.0127781771f,
        -0.0197594142f, +0.0311333953f, -0.0531370391f, +0.1202978923f,
        +0.9795409427f, -0.0952702844f, +0.0465866345f, -0.0280327858f,
        +0.0179290671f, -0.0115915252f, +0.0073891574f, -0.0045709571f,
        +0.0027094601f, -0.0015202727f, +0.0007963882f, -0.0003826515f,
        +0.0001644328f, -0.0000606374f, +0.0000175554f, -0.0000028458f
    },
    {
        -0.0000000247f, +0.0000039282f, -0.0000201014f, +0.0000652124f,
        -0.0001705226f, +0.0003867591f, -0.0007893347f, +0.0014836958f,
        -0.0026117530f, +0.0043631974f, -0.0070020787f, +0.0109361328f,
        -0.0169105828f, +0.0266277894f, -0.0453497634f, +0.1017887543f,
        +0.9849437565f, -0.0833608751f, +0.0405147043f, -0.0243381137f,
        +0.0155587475f, -0.0100596584f, +0.0064149731f, -0.0039706245f,
        +0.0023554255f, -0.0013229101f, +0.0006938430f, -0.0003338946f,
        +0.0001437731f, -0.0000531733f, +0.0000154743f, -0.0000025508f
    },
    {
        -0.0000000143f, +0.0000031800f, -0.0000164749f, +0.0000536885f,
        -0.0001407672f, +0.0003198851f, -0.0006538126f, +0.0012303887f,
        -0.0021678749f, +0.0036243168f, -0.0058194706f, +0.0090919379f,
        -0.0140586566f, +0.0221233294f, -0.0375990419f, +0.0836891719f,
        +0.9895295048f, -0.0708683441f, +0.0342281032f, -0.0205263975f,
        +0.0131157041f, -0.0084805357f, +0.0054099152f, -0.0033504630f,
        +0.0019890651f, -0.0011182288f, +0.0005872016f, -0.0002830104f,
        +0.0001221095f, -0.0000452920f, +0.0000132482f, -0.0000022196f
    },
    {
        -0.0000000074f, +0.0000024683f, -0.0000129504f, +0.0000423952f,
        -0.0001114583f, +0.0002537715f, -0.0005194492f, +0.0009786769f,
        -0.0017259861f, +0.0028876843f, -0.0046392058f, +0.0072502774f,
        -0.0112108219f, +0.0176311308f, -0.0299026598f, +0.0660190952f,
        +0.9932911391f, -0.0578013437f, +0.0277383040f, -0.0166054843f,
        +0.0106051395f, -0.0068575422f, +0.0043761229f, -0.0027117704f,
        +0.0016111255f, -0.0009066304f, +0.0004766622f, -0.0002300860f,
        +0.0000994745f, -0.0000370024f, +0.0000108782f, -0.0000018517f
    },
    {
        -0.0000000031f, +0.0000017939f, -0.0000095346f, +0.0000313572f,
        -0.0000826639f, +0.0001885769f, -0.0003865737f, +0.0007291846f,
        -0.0012871910f, +0.0021551486f, -0.0034642492f, +0.0054157663f,
        -0.0083741567f, +0.0131621155f, -0.0222779562f, +0.0487975515f,
        +0.9962228743f, -0.0441696258f, +0.0210574590f, -0.0125836072f,
        +0.0080324992f, -0.0051942225f, +0.0033158410f, -0.0020559131f,
        +0.0012223968f, -0.0006885427f, +0.0003624375f, -0.0001752162f,
        +0.0000759042f, -0.0000283155f, +0.0000083660f, -0.0000014462f
    },
    {
        -0.0000000009f, +0.0000011575f, -0.0000062340f, +0.0000205978f,
        -0.0000544490f, +0.0001244539f, -0.0002555043f, +0.0004825188f,
        -0.0008525675f, +0.0014285192f, -0.0022975099f, +0.0035929384f,
        -0.0055556148f, +0.0087269885f, -0.0147417944f, +0.0320426187f,
        +0.9983202001f, -0.0299840343f, +0.0141983828f, -0.0084693708f,
        +0.0054034615f, -0.0034942741f, +0.0022314159f, -0.0013843243f,
        +0.0008237112f, -0.0004644190f, +0.0002447554f, -0.0001185038f,
        +0.0000514387f, -0.0000192432f, +0.0000057136f, -0.0000010026f
    },
    {
        -0.0000000001f, +0.0000005594f, -0.0000030541f, +0.0000101387f,
        -0.0000268750f, +0.0000615485f, -0.0001265481f, +0.0002392678f,
        -0.0004231645f, +0.0007095631f, -0.0011418343f, +0.0017862360f,
        -0.0027620098f, +0.0043362153f, -0.0073105332f, +0.0157713998f,
        +0.9995798883f, -0.0152564942f, +0.0071745349f, -0.0042717353f,
        +0.0027239256f, -0.0017615398f, +0.0011252905f, -0.0006985008f,
        +0.0004159416f, -0.0002347374f, +0.0001238576f, -0.0000600592f,
        +0.0000261216f, -0.0000097996f, +0.0000029239f, -0.0000005207f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    }
};

_Alignas(64) static float const kernels_sinc_8_cubic [32][8] =
{
    {
        +0.0002597675f, -0.0039207942f, +0.0252813482f, +0.9976142235f,
        -0.0224072833f, +0.0033759653f, -0.0002028858f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0002028858f, +0.0033759653f, -0.0224072833f, +0.9976142235f,
        +0.0252813482f, -0.0039207942f, +0.0002597675f, -0.0000000762f
    },
    {
        -0.0003549887f, +0.0062188612f, -0.0419547282f, +0.9904817710f,
        +0.0534170529f, -0.0083886329f, +0.0005820920f, -0.0000006154f
    },
    {
        -0.0004626033f, +0.0085486928f, -0.0586884681f, +0.9786769174f,
        +0.0843518729f, -0.0133949381f, +0.0009719698f, -0.0000020958f
    },
    {
        -0.0005320226f, +0.0103923915f, -0.0726841373f, +0.9623222726f,
        +0.1179939134f, -0.0189191675f, +0.0014334389f, -0.0000050129f
    },
    {
        -0.0005693707f, +0.0117825051f, -0.0840441436f, +0.9415870311f,
        +0.1542140211f, -0.0249277566f, +0.0019692793f, -0.0000098768f
    },
    {
        -0.0005804673f, +0.0127559191f, -0.0928946799f, +0.9166845637f,
        +0.1928457270f, -0.0313732031f, +0.0025806987f, -0.0000172073f
    },
    {
        -0.0005707216f, +0.0133526329f, -0.0993825376f, +0.8878693936f,
        +0.2336857581f, -0.0381933275f, +0.0032670113f, -0.0000275246f
    },
    {
        -0.0005450558f, +0.0136146100f, -0.1036717835f, +0.8554336095f,
        +0.2764951337f, -0.0453107385f, +0.0040253171f, -0.0000413342f
    },
    {
        -0.0005078543f, +0.0135847166f, -0.1059403649f, +0.8197027759f,
        +0.3210008449f, -0.0526325333f, +0.0048501897f, -0.0000591070f
    },
    {
        -0.0004629377f, +0.0133057627f, -0.1063767014f, +0.7810314124f,
        +0.3668981130f, -0.0600502573f, +0.0057333854f, -0.0000812528f
    },
    {
        -0.0004135573f, +0.0128196536f, -0.1051763240f, +0.7397981173f,
        +0.4138532056f, -0.0674401466f, +0.0066635808f, -0.0001080870f
    },
    {
        -0.0003624081f, +0.0121666586f, -0.1025386141f, +0.6964004181f,
        +0.4615067821f, -0.0746636711f, +0.0076261527f, -0.0001397917f
    },
    {
        -0.0003116559f, +0.0113847990f, -0.0986636928f, +0.6512494350f,
        +0.5094777289f, -0.0815683908f, +0.0086030112f, -0.0001763709f
    },
    {
        -0.0002629760f, +0.0105093563f, -0.0937495048f, +0.6047644449f,
        +0.5573674344f, -0.0879891352f, +0.0095724973f, -0.0002176007f
    },
    {
        -0.0002176007f, +0.0095724973f, -0.0879891352f, +0.5573674344f,
        +0.6047644449f, -0.0937495048f, +0.0105093563f, -0.0002629760f
    },
    {
        -0.0001763709f, +0.0086030112f, -0.0815683908f, +0.5094777289f,
        +0.6512494350f, -0.0986636928f, +0.0113847990f, -0.0003116559f
    },
    {
        -0.0001397917f, +0.0076261527f, -0.0746636711f, +0.4615067821f,
        +0.6964004181f, -0.1025386141f, +0.0121666586f, -0.0003624081f
    },
    {
        -0.0001080870f, +0.0066635808f, -0.0674401466f, +0.4138532056f,
        +0.7397981173f, -0.1051763240f, +0.0128196536f, -0.0004135573f
    },
    {
        -0.0000812528f, +0.0057333854f, -0.0600502573f, +0.3668981130f,
        +0.7810314124f, -0.1063767014f, +0.0133057627f, -0.0004629377f
    },
    {
        -0.0000591070f, +0.0048501897f, -0.0526325333f, +0.3210008449f,
        +0.8197027759f, -0.1059403649f, +0.0135847166f, -0.0005078543f
    },
    {
        -0.0000413342f, +0.0040253171f, -0.0453107385f, +0.2764951337f,
        +0.8554336095f, -0.1036717835f, +0.0136146100f, -0.0005450558f
    },
    {
        -0.0000275246f, +0.0032670113f, -0.0381933275f, +0.2336857581f,
        +0.8878693936f, -0.0993825376f, +0.0133526329f, -0.0005707216f
    },
    {
        -0.0000172073f, +0.0025806987f, -0.0313732031f, +0.1928457270f,
        +0.9166845637f, -0.0928946799f, +0.0127559191f, -0.0005804673f
    },
    {
        -0.0000098768f, +0.0019692793f, -0.0249277566f, +0.1542140211f,
        +0.9415870311f, -0.0840441436f, +0.0117825051f, -0.0005693707f
    },
    {
        -0.0000050129f, +0.0014334389f, -0.0189191675f, +0.1179939134f,
        +0.9623222726f, -0.0726841373f, +0.0103923915f, -0.0005320226f
    },
    {
        -0.0000020958f, +0.0009719698f, -0.0133949381f, +0.0843518729f,
        +0.9786769174f, -0.0586884681f, +0.0085486928f, -0.0004626033f
    },
    {
        -0.0000006154f, +0.0005820920f, -0.0083886329f, +0.0534170529f,
        +0.9904817710f, -0.0419547282f, +0.0062188612f, -0.0003549887f
    },
    {
        -0.0000000762f, +0.0002597675f, -0.0039207942f, +0.0252813482f,
        +0.9976142235f, -0.0224072833f, +0.0033759653f, -0.0002028858f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0002028858f, +0.0033759653f, -0.0224072833f,
        +0.9976142235f, +0.0252813482f, -0.0039207942f, +0.0002597675f
    }
};

_Alignas(64) static float const kernels_sinc_16_cubic [32][16] =
{
    {
        +0.0000142372f, -0.0001222265f, +0.0005689651f, -0.0018891417f,
        +0.0050754959f, -0.0122638191f, +0.0327412870f, +0.9979374556f,
        -0.0301752646f, +0.0115458264f, -0.0047636357f, +0.0017529852f,
        -0.0005182446f, +0.0001080204f, -0.0000117807f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000117807f, +0.0001080204f, -0.0005182446f, +0.0017529852f,
        -0.0047636357f, +0.0115458264f, -0.0301752646f, +0.9979374556f,
        +0.0327412870f, -0.0122638191f, +0.0050754959f, -0.0018891417f,
        +0.0005689651f, -0.0001222265f, +0.0000142372f, -0.0000000095f
    },
    {
        -0.0000212377f, +0.0002016965f, -0.0009828203f, +0.0033561752f,
        -0.0091739032f, +0.0222759309f, -0.0576769779f, +0.9917661662f,
        +0.0679146713f, -0.0251331479f, +0.0104144883f, -0.0038978103f,
        +0.0011846264f, -0.0002582448f, +0.0000310269f, -0.0000000758f
    },
    {
        -0.0000285336f, +0.0002811408f, -0.0013918123f, +0.0047989121f,
        -0.0131962971f, +0.0321077070f, -0.0824243242f, +0.9815349991f,
        +0.1053604338f, -0.0384813103f, +0.0159620092f, -0.0060064202f,
        +0.0018418555f, -0.0004073410f, +0.0000504203f, -0.0000002550f
    },
    {
        -0.0000338536f, +0.0003466893f, -0.0017442842f, +0.0060734677f,
        -0.0168033627f, +0.0409738500f, -0.1043632345f, +0.9673248576f,
        +0.1448939482f, -0.0521681919f, +0.0216568422f, -0.0081925113f,
        +0.0025344075f, -0.0005684830f, +0.0000724164f, -0.0000006010f
    },
    {
        -0.0000373982f, +0.0003988727f, -0.0020402120f, +0.0071749533f,
        -0.0199746479f, +0.0488223507f, -0.1234661426f, +0.9492478724f,
        +0.1863068468f, -0.0660411847f, +0.0274319851f, -0.0104308793f,
        +0.0032549260f, -0.0007403036f, +0.0000969544f, -0.0000011634f
    },
    {
        -0.0000393780f, +0.0004383875f, -0.0022804092f, +0.0081011958f,
        -0.0226965652f, +0.0556163129f, -0.1397314714f, +0.9274462843f,
        +0.2293684592f, -0.0799363198f, +0.0332152172f, -0.0126937555f,
        +0.0039949671f, -0.0009210902f, +0.0001239070f, -0.0000019861f
    },
    {
        -0.0000400071f, +0.0004660666f, -0.0024664455f, +0.0088525824f,
        -0.0249621715f, +0.0613336021f, -0.1531828621f, +0.9020910290f,
        +0.2738275104f, -0.0936795786f, +0.0389297706f, -0.0149510380f,
        +0.0047450430f, -0.0011087789f, +0.0001530733f, -0.0000031058f
    },
    {
        -0.0000394985f, +0.0004828506f, -0.0026005590f, +0.0094318791f,
        -0.0267708702f, +0.0659663364f, -0.1638681598f, +0.8733800404f,
        +0.3194140663f, -0.1070883754f, +0.0444950995f, -0.0171705714f,
        +0.0054946854f, -0.0013009540f, +0.0001841735f, -0.0000045496f
    },
    {
        -0.0000380595f, +0.0004897602f, -0.0026855661f, +0.0098440277f,
        -0.0281280474f, +0.0695202339f, -0.1718581730f, +0.8415362932f,
        +0.3658417060f, -0.1199731976f, +0.0498277448f, -0.0193184765f,
        +0.0062325316f, -0.0014948552f, +0.0002168439f, -0.0000063340f
    },
    {
        -0.0000358882f, +0.0004878694f, -0.0027247691f, +0.0100959241f,
        -0.0290446470f, +0.0720138275f, -0.1772452254f, +0.8068056046f,
        +0.4128099039f, -0.1321393907f, +0.0548422843f, -0.0213595271f,
        +0.0069464297f, -0.0016873911f, +0.0002506334f, -0.0000084630f
    },
    {
        -0.0000331701f, +0.0004782815f, -0.0027218637f, +0.0101961838f,
        -0.0295366969f, +0.0734775655f, -0.1801415221f, +0.7694542222f,
        +0.4600065953f, -0.1433890707f, +0.0594523625f, -0.0232575705f,
        +0.0076235672f, -0.0018751595f, +0.0002850013f, -0.0000109264f
    },
    {
        -0.0000300758f, +0.0004621060f, -0.0026808482f, +0.0101548987f,
        -0.0296247927f, +0.0739528105f, -0.1806773531f, +0.7297662246f,
        +0.5071109014f, -0.1535231460f, +0.0635717883f, -0.0249759894f,
        +0.0082506182f, -0.0020544763f, +0.0003193171f, -0.0000136986f
    },
    {
        -0.0000267592f, +0.0004404382f, -0.0026059351f, +0.0099833884f,
        -0.0293335506f, +0.0734907557f, -0.1789991574f, +0.6880407631f,
        +0.5537959863f, -0.1623434278f, +0.0671156908f, -0.0264782005f,
        +0.0088139118f, -0.0022214120f, +0.0003528617f, -0.0000167374f
    },
    {
        -0.0000233559f, +0.0004143413f, -0.0025014665f, +0.0096939513f,
        -0.0286910368f, +0.0721512743f, -0.1752674740f, +0.6445891772f,
        +0.5997320140f, -0.1696548069f, +0.0700017192f, -0.0277281836f,
        +0.0092996180f, -0.0023718355f, +0.0003848309f, -0.0000199827f
    },
    {
        -0.0000199827f, +0.0003848309f, -0.0023718355f, +0.0092996180f,
        -0.0277281836f, +0.0700017192f, -0.1696548069f, +0.5997320140f,
        +0.6445891772f, -0.1752674740f, +0.0721512743f, -0.0286910368f,
        +0.0096939513f, -0.0025014665f, +0.0004143413f, -0.0000233559f
    },
    {
        -0.0000167374f, +0.0003528617f, -0.0022214120f, +0.0088139118f,
        -0.0264782005f, +0.0671156908f, -0.1623434278f, +0.5537959863f,
        +0.6880407631f, -0.1789991574f, +0.0734907557f, -0.0293335506f,
        +0.0099833884f, -0.0026059351f, +0.0004404382f, -0.0000267592f
    },
    {
        -0.0000136986f, +0.0003193171f, -0.0020544763f, +0.0082506182f,
        -0.0249759894f, +0.0635717883f, -0.1535231460f, +0.5071109014f,
        +0.7297662246f, -0.1806773531f, +0.0739528105f, -0.0296247927f,
        +0.0101548987f, -0.0026808482f, +0.0004621060f, -0.0000300758f
    },
    {
        -0.0000109264f, +0.0002850013f, -0.0018751595f, +0.0076235672f,
        -0.0232575705f, +0.0594523625f, -0.1433890707f, +0.4600065953f,
        +0.7694542222f, -0.1801415221f, +0.0734775655f, -0.0295366969f,
        +0.0101961838f, -0.0027218637f, +0.0004782815f, -0.0000331701f
    },
    {
        -0.0000084630f, +0.0002506334f, -0.0016873911f, +0.0069464297f,
        -0.0213595271f, +0.0548422843f, -0.1321393907f, +0.4128099039f,
        +0.8068056046f, -0.1772452254f, +0.0720138275f, -0.0290446470f,
        +0.0100959241f, -0.0027247691f, +0.0004878694f, -0.0000358882f
    },
    {
        -0.0000063340f, +0.0002168439f, -0.0014948552f, +0.0062325316f,
        -0.0193184765f, +0.0498277448f, -0.1199731976f, +0.3658417060f,
        +0.8415362932f, -0.1718581730f, +0.0695202339f, -0.0281280474f,
        +0.0098440277f, -0.0026855661f, +0.0004897602f, -0.0000380595f
    },
    {
        -0.0000045496f, +0.0001841735f, -0.0013009540f, +0.0054946854f,
        -0.0171705714f, +0.0444950995f, -0.1070883754f, +0.3194140663f,
        +0.8733800404f, -0.1638681598f, +0.0659663364f, -0.0267708702f,
        +0.0094318791f, -0.0026005590f, +0.0004828506f, -0.0000394985f
    },
    {
        -0.0000031058f, +0.0001530733f, -0.0011087789f, +0.0047450430f,
        -0.0149510380f, +0.0389297706f, -0.0936795786f, +0.2738275104f,
        +0.9020910290f, -0.1531828621f, +0.0613336021f, -0.0249621715f,
        +0.0088525824f, -0.0024664455f, +0.0004660666f, -0.0000400071f
    },
    {
        -0.0000019861f, +0.0001239070f, -0.0009210902f, +0.0039949671f,
        -0.0126937555f, +0.0332152172f, -0.0799363198f, +0.2293684592f,
        +0.9274462843f, -0.1397314714f, +0.0556163129f, -0.0226965652f,
        +0.0081011958f, -0.0022804092f, +0.0004383875f, -0.0000393780f
    },
    {
        -0.0000011634f, +0.0000969544f, -0.0007403036f, +0.0032549260f,
        -0.0104308793f, +0.0274319851f, -0.0660411847f, +0.1863068468f,
        +0.9492478724f, -0.1234661426f, +0.0488223507f, -0.0199746479f,
        +0.0071749533f, -0.0020402120f, +0.0003988727f, -0.0000373982f
    },
    {
        -0.0000006010f, +0.0000724164f, -0.0005684830f, +0.0025344075f,
        -0.0081925113f, +0.0216568422f, -0.0521681919f, +0.1448939482f,
        +0.9673248576f, -0.1043632345f, +0.0409738500f, -0.0168033627f,
        +0.0060734677f, -0.0017442842f, +0.0003466893f, -0.0000338536f
    },
    {
        -0.0000002550f, +0.0000504203f, -0.0004073410f, +0.0018418555f,
        -0.0060064202f, +0.0159620092f, -0.0384813103f, +0.1053604338f,
        +0.9815349991f, -0.0824243242f, +0.0321077070f, -0.0131962971f,
        +0.0047989121f, -0.0013918123f, +0.0002811408f, -0.0000285336f
    },
    {
        -0.0000000758f, +0.0000310269f, -0.0002582448f, +0.0011846264f,
        -0.0038978103f, +0.0104144883f, -0.0251331479f, +0.0679146713f,
        +0.9917661662f, -0.0576769779f, +0.0222759309f, -0.0091739032f,
        +0.0033561752f, -0.0009828203f, +0.0002016965f, -0.0000212377f
    },
    {
        -0.0000000095f, +0.0000142372f, -0.0001222265f, +0.0005689651f,
        -0.0018891417f, +0.0050754959f, -0.0122638191f, +0.0327412870f,
        +0.9979374556f, -0.0301752646f, +0.0115458264f, -0.0047636357f,
        +0.0017529852f, -0.0005182446f, +0.0001080204f, -0.0000117807f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000117807f, +0.0001080204f, -0.0005182446f,
        +0.0017529852f, -0.0047636357f, +0.0115458264f, -0.0301752646f,
        +0.9979374556f, +0.0327412870f, -0.0122638191f, +0.0050754959f,
        -0.0018891417f, +0.0005689651f, -0.0001222265f, +0.0000142372f
    }
};

_Alignas(64) static float const kernels_sinc_24_cubic [32][24] =
{
    {
        +0.0000033144f, -0.0000209910f, +0.0000798387f, -0.0002338727f,
        +0.0005748807f, -0.0012439113f, +0.0024495327f, -0.0045199995f,
        +0.0080939427f, -0.0149662792f, +0.0343262200f, +0.9979973240f,
        -0.0318593467f, +0.0142966758f, -0.0077748779f, +0.0043392521f,
        -0.0023433708f, +0.0011834015f, -0.0005427604f, +0.0002185407f,
        -0.0000735260f, +0.0000188839f, -0.0000028063f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000028063f, +0.0000188839f, -0.0000735260f, +0.0002185407f,
        -0.0005427604f, +0.0011834015f, -0.0023433708f, +0.0043392521f,
        -0.0077748779f, +0.0142966758f, -0.0318593467f, +0.9979973240f,
        +0.0343262200f, -0.0149662792f, +0.0080939427f, -0.0045199995f,
        +0.0024495327f, -0.0012439113f, +0.0005748807f, -0.0002338727f,
        +0.0000798387f, -0.0000209910f, +0.0000033144f, -0.0000000028f
    },
    {
        -0.0000051172f, +0.0000355820f, -0.0001402267f, +0.0004199031f,
        -0.0010483357f, +0.0022945952f, -0.0045566386f, +0.0084528537f,
        -0.0151514077f, +0.0277905079f, -0.0611219134f, +0.9920041847f,
        +0.0709654097f, -0.0304553129f, +0.0164205964f, -0.0091717408f,
        +0.0049788618f, -0.0025352555f, +0.0011760909f, -0.0004808896f,
        +0.0001653406f, -0.0000439673f, +0.0000071402f, -0.0000000224f
    },
    {
        -0.0000069535f, +0.0000500567f, -0.0001997236f, +0.0006025858f,
        -0.0015124149f, +0.0033233465f, -0.0066184825f, +0.0123003897f,
        -0.0220580368f, +0.0403627372f, -0.0876824355f, +0.9820651127f,
        +0.1097405537f, -0.0463076883f, +0.0248873056f, -0.0139025690f,
        +0.0075594333f, -0.0038596835f, +0.0017971530f, -0.0007385310f,
        +0.0002557187f, -0.0000687613f, +0.0000114700f, -0.0000000750f
    },
    {
        -0.0000083432f, +0.0000623081f, -0.0002517679f, +0.0007654320f,
        -0.0019314560f, +0.0042609450f, -0.0085103569f, +0.0158462501f,
        -0.0284314902f, +0.0519097820f, -0.1114606930f, +0.9682538750f,
        +0.1504526450f, -0.0623526392f, +0.0333959862f, -0.0186566958f,
        +0.0101607794f, -0.0052017031f, +0.0024309532f, -0.0010039595f,
        +0.0003500502f, -0.0000951590f, +0.0000162848f, -0.0000001762f
    },
    {
        -0.0000093200f, +0.0000723707f, -0.0002962347f, +0.0009076231f,
        -0.0023026917f, +0.0051002440f, -0.0102166050f, +0.0190598760f,
        -0.0342172021f, +0.0623437547f, -0.1324016689f, +0.9506728031f,
        +0.1928820316f, -0.0784094115f, +0.0418440948f, -0.0233757758f,
        +0.0127508293f, -0.0065448293f, +0.0030698041f, -0.0012740408f,
        +0.0004472797f, -0.0001228985f, +0.0000215531f, -0.0000003398f
    },
    {
        -0.0000099218f, +0.0000803100f, -0.0003331156f, +0.0010286690f,
        -0.0026241236f, +0.0058356775f, -0.0117245319f, +0.0219159352f,
        -0.0393696295f, +0.0715927976f, -0.1504754667f, +0.9294518630f,
        +0.2367899788f, -0.0942887757f, +0.0501256871f, -0.0279995362f,
        +0.0152962545f, -0.0078717557f, +0.0037055186f, -0.0015453689f,
        +0.0005462256f, -0.0001516701f, +0.0000272302f, -0.0000005775f
    },
    {
        -0.0000101901f, +0.0000862186f, -0.0003625099f, +0.0011283939f,
        -0.0028945063f, +0.0064632554f, -0.0130244366f, +0.0243944269f,
        -0.0438524507f, +0.0796012403f, -0.1656769911f, +0.9047474747f,
        +0.2819204347f, -0.1097946714f, +0.0581325547f, -0.0324664563f,
        +0.0177628446f, -0.0091645441f, +0.0043294942f, -0.0018142982f,
        +0.0006455881f, -0.0001811176f, +0.0000332571f, -0.0000008984f
    },
    {
        -0.0000101683f, +0.0000902122f, -0.0003846154f, +0.0012069187f,
        -0.0031133231f, +0.0069805397f, -0.0141096059f, +0.0264807174f,
        -0.0476386478f, +0.0863295806f, -0.1780253982f, +0.8767410952f,
        +0.3280019845f, -0.1247259707f, +0.0657554308f, -0.0367144873f,
        +0.0201159107f, -0.0104048325f, +0.0049328089f, -0.0020769795f,
        +0.0007439606f, -0.0002108394f, +0.0000395610f, -0.0000013087f
    },
    {
        -0.0000099007f, +0.0000924257f, -0.0003997180f, +0.0012646409f,
        -0.0032807525f, +0.0073866010f, -0.0149762701f, +0.0281655088f,
        -0.0507104777f, +0.0917542956f, -0.1875633263f, +0.8456375775f,
        +0.3747499743f, -0.1388783446f, +0.0728852517f, -0.0406818075f,
        +0.0223207119f, -0.0115740576f, +0.0055063250f, -0.0023294028f,
        +0.0008398422f, -0.0002403912f, +0.0000460543f, -0.0000018109f
    },
    {
        -0.0000094318f, +0.0000930092f, -0.0004081806f, +0.0013022106f,
        -0.0033976276f, +0.0076819573f, -0.0156235240f, +0.0294447433f,
        -0.0530593343f, +0.0958674873f, -0.1943559165f, +0.8116633236f,
        +0.4218687880f, -0.1520462140f, +0.0794144622f, -0.0443076031f,
        +0.0243428992f, -0.0126536909f, +0.0060408025f, -0.0025674435f,
        +0.0009316541f, -0.0002692897f, +0.0000526353f, -0.0000024037f
    },
    {
        -0.0000088047f, +0.0000921243f, -0.0004104323f, +0.0013205063f,
        -0.0034653890f, +0.0078684977f, -0.0160532150f, +0.0303194471f,
        -0.0546855091f, +0.0986763736f, -0.1984896390f, +0.7750642500f,
        +0.4690542534f, -0.1640247678f, +0.0852383507f, -0.0475328662f,
        +0.0261489746f, -0.0136254854f, +0.0065270196f, -0.0027869147f,
        +0.0010177571f, -0.0002970167f, +0.0000591884f, -0.0000030815f
    },
    {
        -0.0000080611f, +0.0000899403f, -0.0004069570f, +0.0013206070f,
        -0.0034860316f, +0.0079493917f, -0.0162698005f, +0.0307955176f,
        -0.0555978548f, +0.1002026338f, -0.2000709375f, +0.7361035848f,
        +0.5159961578f, -0.1746120273f, +0.0902563986f, -0.0503012008f,
        +0.0277067564f, -0.0144717289f, +0.0069558992f, -0.0029836220f,
        +0.0010964721f, -0.0003230251f, +0.0000655847f, -0.0000038345f
    },
    {
        -0.0000072398f, +0.0000866310f, -0.0003982815f, +0.0013037653f,
        -0.0034620479f, +0.0079289853f, -0.0162801796f, +0.0308834592f,
        -0.0558133600f, +0.1004816186f, -0.1992247116f, +0.6950595211f,
        +0.5623808472f, -0.1836109370f, +0.0943736309f, -0.0525596267f,
        +0.0289858477f, -0.0151755034f, +0.0073186402f, -0.0031534228f,
        +0.0011661023f, -0.0003467444f, +0.0000716832f, -0.0000046481f
    },
    {
        -0.0000063765f, +0.0000823707f, -0.0003849653f, +0.0012713778f,
        -0.0033963663f, +0.0078126882f, -0.0160934999f, +0.0305980721f,
        -0.0553566427f, +0.0995614398f, -0.1960926535f, +0.6522227481f,
        +0.6078938857f, -0.1908314595f, +0.0975019503f, -0.0542593720f,
        +0.0299581009f, -0.0157209447f, +0.0076068516f, -0.0032922882f,
        +0.0012249570f, -0.0003675889f, +0.0000773321f, -0.0000055033f
    },
    {
        -0.0000055033f, +0.0000773321f, -0.0003675889f, +0.0012249570f,
        -0.0032922882f, +0.0076068516f, -0.0157209447f, +0.0299581009f,
        -0.0542593720f, +0.0975019503f, -0.1908314595f, +0.6078938857f,
        +0.6522227481f, -0.1960926535f, +0.0995614398f, -0.0553566427f,
        +0.0305980721f, -0.0160934999f, +0.0078126882f, -0.0033963663f,
        +0.0012713778f, -0.0003849653f, +0.0000823707f, -0.0000063765f
    },
    {
        -0.0000046481f, +0.0000716832f, -0.0003467444f, +0.0011661023f,
        -0.0031534228f, +0.0073186402f, -0.0151755034f, +0.0289858477f,
        -0.0525596267f, +0.0943736309f, -0.1836109370f, +0.5623808472f,
        +0.6950595211f, -0.1992247116f, +0.1004816186f, -0.0558133600f,
        +0.0308834592f, -0.0162801796f, +0.0079289853f, -0.0034620479f,
        +0.0013037653f, -0.0003982815f, +0.0000866310f, -0.0000072398f
    },
    {
        -0.0000038345f, +0.0000655847f, -0.0003230251f, +0.0010964721f,
        -0.0029836220f, +0.0069558992f, -0.0144717289f, +0.0277067564f,
        -0.0503012008f, +0.0902563986f, -0.1746120273f, +0.5159961578f,
        +0.7361035848f, -0.2000709375f, +0.1002026338f, -0.0555978548f,
        +0.0307955176f, -0.0162698005f, +0.0079493917f, -0.0034860316f,
        +0.0013206070f, -0.0004069570f, +0.0000899403f, -0.0000080611f
    },
    {
        -0.0000030815f, +0.0000591884f, -0.0002970167f, +0.0010177571f,
        -0.0027869147f, +0.0065270196f, -0.0136254854f, +0.0261489746f,
        -0.0475328662f, +0.0852383507f, -0.1640247678f, +0.4690542534f,
        +0.7750642500f, -0.1984896390f, +0.0986763736f, -0.0546855091f,
        +0.0303194471f, -0.0160532150f, +0.0078684977f, -0.0034653890f,
        +0.0013205063f, -0.0004104323f, +0.0000921243f, -0.0000088047f
    },
    {
        -0.0000024037f, +0.0000526353f, -0.0002692897f, +0.0009316541f,
        -0.0025674435f, +0.0060408025f, -0.0126536909f, +0.0243428992f,
        -0.0443076031f, +0.0794144622f, -0.1520462140f, +0.4218687880f,
        +0.8116633236f, -0.1943559165f, +0.0958674873f, -0.0530593343f,
        +0.0294447433f, -0.0156235240f, +0.0076819573f, -0.0033976276f,
        +0.0013022106f, -0.0004081806f, +0.0000930092f, -0.0000094318f
    },
    {
        -0.0000018109f, +0.0000460543f, -0.0002403912f, +0.0008398422f,
        -0.0023294028f, +0.0055063250f, -0.0115740576f, +0.0223207119f,
        -0.0406818075f, +0.0728852517f, -0.1388783446f, +0.3747499743f,
        +0.8456375775f, -0.1875633263f, +0.0917542956f, -0.0507104777f,
        +0.0281655088f, -0.0149762701f, +0.0073866010f, -0.0032807525f,
        +0.0012646409f, -0.0003997180f, +0.0000924257f, -0.0000099007f
    },
    {
        -0.0000013087f, +0.0000395610f, -0.0002108394f, +0.0007439606f,
        -0.0020769795f, +0.0049328089f, -0.0104048325f, +0.0201159107f,
        -0.0367144873f, +0.0657554308f, -0.1247259707f, +0.3280019845f,
        +0.8767410952f, -0.1780253982f, +0.0863295806f, -0.0476386478f,
        +0.0264807174f, -0.0141096059f, +0.0069805397f, -0.0031133231f,
        +0.0012069187f, -0.0003846154f, +0.0000902122f, -0.0000101683f
    },
    {
        -0.0000008984f, +0.0000332571f, -0.0001811176f, +0.0006455881f,
        -0.0018142982f, +0.0043294942f, -0.0091645441f, +0.0177628446f,
        -0.0324664563f, +0.0581325547f, -0.1097946714f, +0.2819204347f,
        +0.9047474747f, -0.1656769911f, +0.0796012403f, -0.0438524507f,
        +0.0243944269f, -0.0130244366f, +0.0064632554f, -0.0028945063f,
        +0.0011283939f, -0.0003625099f, +0.0000862186f, -0.0000101901f
    },
    {
        -0.0000005775f, +0.0000272302f, -0.0001516701f, +0.0005462256f,
        -0.0015453689f, +0.0037055186f, -0.0078717557f, +0.0152962545f,
        -0.0279995362f, +0.0501256871f, -0.0942887757f, +0.2367899788f,
        +0.9294518630f, -0.1504754667f, +0.0715927976f, -0.0393696295f,
        +0.0219159352f, -0.0117245319f, +0.0058356775f, -0.0026241236f,
        +0.0010286690f, -0.0003331156f, +0.0000803100f, -0.0000099218f
    },
    {
        -0.0000003398f, +0.0000215531f, -0.0001228985f, +0.0004472797f,
        -0.0012740408f, +0.0030698041f, -0.0065448293f, +0.0127508293f,
        -0.0233757758f, +0.0418440948f, -0.0784094115f, +0.1928820316f,
        +0.9506728031f, -0.1324016689f, +0.0623437547f, -0.0342172021f,
        +0.0190598760f, -0.0102166050f, +0.0051002440f, -0.0023026917f,
        +0.0009076231f, -0.0002962347f, +0.0000723707f, -0.0000093200f
    },
    {
        -0.0000001762f, +0.0000162848f, -0.0000951590f, +0.0003500502f,
        -0.0010039595f, +0.0024309532f, -0.0052017031f, +0.0101607794f,
        -0.0186566958f, +0.0333959862f, -0.0623526392f, +0.1504526450f,
        +0.9682538750f, -0.1114606930f, +0.0519097820f, -0.0284314902f,
        +0.0158462501f, -0.0085103569f, +0.0042609450f, -0.0019314560f,
        +0.0007654320f, -0.0002517679f, +0.0000623081f, -0.0000083432f
    },
    {
        -0.0000000750f, +0.0000114700f, -0.0000687613f, +0.0002557187f,
        -0.0007385310f, +0.0017971530f, -0.0038596835f, +0.0075594333f,
        -0.0139025690f, +0.0248873056f, -0.0463076883f, +0.1097405537f,
        +0.9820651127f, -0.0876824355f, +0.0403627372f, -0.0220580368f,
        +0.0123003897f, -0.0066184825f, +0.0033233465f, -0.0015124149f,
        +0.0006025858f, -0.0001997236f, +0.0000500567f, -0.0000069535f
    },
    {
        -0.0000000224f, +0.0000071402f, -0.0000439673f, +0.0001653406f,
        -0.0004808896f, +0.0011760909f, -0.0025352555f, +0.0049788618f,
        -0.0091717408f, +0.0164205964f, -0.0304553129f, +0.0709654097f,
        +0.9920041847f, -0.0611219134f, +0.0277905079f, -0.0151514077f,
        +0.0084528537f, -0.0045566386f, +0.0022945952f, -0.0010483357f,
        +0.0004199031f, -0.0001402267f, +0.0000355820f, -0.0000051172f
    },
    {
        -0.0000000028f, +0.0000033144f, -0.0000209910f, +0.0000798387f,
        -0.0002338727f, +0.0005748807f, -0.0012439113f, +0.0024495327f,
        -0.0045199995f, +0.0080939427f, -0.0149662792f, +0.0343262200f,
        +0.9979973240f, -0.0318593467f, +0.0142966758f, -0.0077748779f,
        +0.0043392521f, -0.0023433708f, +0.0011834015f, -0.0005427604f,
        +0.0002185407f, -0.0000735260f, +0.0000188839f, -0.0000028063f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000028063f, +0.0000188839f, -0.0000735260f,
        +0.0002185407f, -0.0005427604f, +0.0011834015f, -0.0023433708f,
        +0.0043392521f, -0.0077748779f, +0.0142966758f, -0.0318593467f,
        +0.9979973240f, +0.0343262200f, -0.0149662792f, +0.0080939427f,
        -0.0045199995f, +0.0024495327f, -0.0012439113f, +0.0005748807f,
        -0.0002338727f, +0.0000798387f, -0.0000209910f, +0.0000033144f
    }
};

_Alignas(64) static float const kernels_sinc_32_cubic [32][32] =
{
    {
        +0.0000012645f, -0.0000067946f, +0.0000224320f, -0.0000592696f,
        +0.0001354267f, -0.0002779601f, +0.0005248199f, -0.0009271590f,
        +0.0015533014f, -0.0024979610f, +0.0039061912f, -0.0060399793f,
        +0.0094887898f, -0.0160340981f, +0.0348973545f, +0.9980182787f,
        -0.0324688367f, +0.0153931209f, -0.0091849261f, +0.0058605138f,
        -0.0037898167f, +0.0024200032f, -0.0015011736f, +0.0008931230f,
        -0.0005034712f, +0.0002652818f, -0.0001284086f, +0.0000557190f,
        -0.0000208343f, +0.0000061808f, -0.0000010818f,  0.0000000000f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f, +1.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
        -0.0000010818f, +0.0000061808f, -0.0000208343f, +0.0000557190f,
        -0.0001284086f, +0.0002652818f, -0.0005034712f, +0.0008931230f,
        -0.0015011736f, +0.0024200032f, -0.0037898167f, +0.0058605138f,
        -0.0091849261f, +0.0153931209f, -0.0324688367f, +0.9980182787f,
        +0.0348973545f, -0.0160340981f, +0.0094887898f, -0.0060399793f,
        +0.0039061912f, -0.0024979610f, +0.0015533014f, -0.0009271590f,
        +0.0005248199f, -0.0002779601f, +0.0001354267f, -0.0000592696f,
        +0.0000224320f, -0.0000067946f, +0.0000012645f, -0.0000000012f
    },
    {
        -0.0000019828f, +0.0000117116f, -0.0000399064f, +0.0001073872f,
        -0.0002485605f, +0.0005152066f, -0.0009803506f, +0.0017427091f,
        -0.0029340187f, +0.0047357111f, -0.0074219332f, +0.0114781442f,
        -0.0179693531f, +0.0299983759f, -0.0623715136f, +0.9920875035f,
        +0.0720625023f, -0.0325493562f, +0.0191780947f, -0.0121919281f,
        +0.0078847572f, -0.0050457431f, +0.0031413256f, -0.0018780675f,
        +0.0010652546f, -0.0005656298f, +0.0002764734f, -0.0001215100f,
        +0.0000462618f, -0.0000141538f, +0.0000027095f, -0.0000000094f
    },
    {
        -0.0000027078f, +0.0000165689f, -0.0000570867f, +0.0001545864f,
        -0.0003593845f, +0.0007474069f, -0.0014259322f, +0.0025401311f,
        -0.0042836961f, +0.0069228469f, -0.0108581105f, +0.0167941732f,
        -0.0262642439f, +0.0436828569f, -0.0895942203f, +0.9822507145f,
        +0.1113126084f, -0.0493742406f, +0.0289588411f, -0.0183849603f,
        +0.0118894812f, -0.0076137015f, +0.0047456404f, -0.0028417551f,
        +0.0016151325f, -0.0008597757f, +0.0004215934f, -0.0001860652f,
        +0.0000712550f, -0.0000220148f, +0.0000043292f, -0.0000000316f
    },
    {
        -0.0000032650f, +0.0000207412f, -0.0000722804f, +0.0001969809f,
        -0.0004599821f, +0.0009598425f, -0.0018360835f, +0.0032777097f,
        -0.0055368629f, +0.0089594224f, -0.0140635597f, +0.0217551320f,
        -0.0339889340f, +0.0563285222f, -0.1140474994f, +0.9685792243f,
        +0.1524439164f, -0.0663268640f, +0.0387173115f, -0.0245455488f,
        +0.0158724270f, -0.0101710034f, +0.0063469827f, -0.0038066817f,
        +0.0021679066f, -0.0011569305f, +0.0005691040f, -0.0002522071f,
        +0.0000971433f, -0.0000303015f, +0.0000061130f, -0.0000000740f
    },
    {
        -0.0000036648f, +0.0000242282f, -0.0000854264f, +0.0002343176f,
        -0.0005496306f, +0.0011508111f, -0.0022072742f, +0.0039487735f,
        -0.0066817850f, +0.0108259326f, -0.0170072549f, +0.0263132746f,
        -0.0410717922f, +0.0678329093f, -0.1356665369f, +0.9511719921f,
        +0.1952331456f, -0.0832164873f, +0.0483363121f, -0.0305983271f,
        +0.0197844725f, -0.0126859712f, +0.0079254716f, -0.0047608622f,
        +0.0027167232f, -0.0014534276f, +0.0007172029f, -0.0003191422f,
        +0.0001236264f, -0.0000389244f, +0.0000080461f, -0.0000001425f
    },
    {
        -0.0000039198f, +0.0000270400f, -0.0000964957f, +0.0002664240f,
        -0.0006277834f, +0.0013189554f, -0.0025365958f, +0.0045477043f,
        -0.0077084263f, +0.0125055128f, -0.0196621911f, +0.0304269704f,
        -0.0474507593f, +0.0781096722f, -0.1544112244f, +0.9301547546f,
        +0.2394390841f, -0.0998451493f, +0.0576964130f, -0.0364669358f,
        +0.0235758735f, -0.0151264486f, +0.0094608367f, -0.0056920032f,
        +0.0032544977f, -0.0017454402f, +0.0008639854f, -0.0003860180f,
        +0.0001503745f, -0.0000477810f, +0.0000101091f, -0.0000002417f
    },
    {
        -0.0000040443f, +0.0000291962f, -0.0001054898f, +0.0002932065f,
        -0.0006940681f, +0.0014632641f, -0.0028217731f, +0.0050699679f,
        -0.0086085142f, +0.0139840604f, -0.0220055878f, +0.0340610154f,
        -0.0530737661f, +0.0870889414f, -0.1702659926f, +0.9056789241f,
        +0.2848043664f, -0.1160094089f, +0.0666772507f, -0.0420749059f,
        +0.0271968506f, -0.0174601824f, +0.0109326585f, -0.0065876465f,
        +0.0037739967f, -0.0020290245f, +0.0010074656f, -0.0004519321f,
        +0.0001770302f, -0.0000567561f, +0.0000122775f, -0.0000003753f
    },
    {
        -0.0000040537f, +0.0000307247f, -0.0001124389f, +0.0003146466f,
        -0.0007482824f, +0.0015830695f, -0.0030611663f, +0.0055121300f,
        -0.0093755812f, +0.0152503190f, -0.0240190360f, +0.0371868591f,
        -0.0578990264f, +0.0947175064f, -0.1832394230f, +0.8779202636f,
        +0.3310574185f, -0.1315021858f, +0.0751588798f, -0.0473465686f,
        +0.0305981951f, -0.0196552177f, +0.0123206187f, -0.0074353219f,
        +0.0042679248f, -0.0023001659f, +0.0011455986f, -0.0005159412f,
        +0.0002032129f, -0.0000657233f, +0.0000145225f, -0.0000005454f
    },
    {
        -0.0000039641f, +0.0000316605f, -0.0001173988f, +0.0003307972f,
        -0.0007903882f, +0.0016780403f, -0.0032537663f, +0.0058718582f,
        -0.0100049828f, +0.0162959253f, -0.0256885889f, +0.0397827472f,
        -0.0618952059f, +0.1009588230f, -0.1933636434f, +0.8470773515f,
        +0.3779145552f, -0.1461146806f, +0.0830231596f, -0.0522079819f,
        +0.0337318847f, -0.0216803022f, +0.0136047578f, -0.0082227040f,
        +0.0047290161f, -0.0025548284f, +0.0012763049f, -0.0005770723f,
        +0.0002285221f, -0.0000745454f, +0.0000168103f, -0.0000007529f
    },
    {
        -0.0000037923f, +0.0000320450f, -0.0001204491f, +0.0003417769f,
        -0.0008205037f, +0.0017481703f, -0.0033991812f, +0.0061479103f,
        -0.0104938934f, +0.0171154204f, -0.0270047965f, +0.0418337815f,
        -0.0650414694f, +0.1057928503f, -0.2006935166f, +0.8133698514f,
        +0.4250822109f, -0.1596383586f, +0.0901551619f, -0.0565878630f,
        +0.0365517032f, -0.0235052933f, +0.0147657362f, -0.0089377738f,
        +0.0051501286f, -0.0027890067f, +0.0013974965f, -0.0006343344f,
        +0.0002525421f, -0.0000830761f, +0.0000191031f, -0.0000009968f
    },
    {
        -0.0000035546f, +0.0000319243f, -0.0001216900f, +0.0003477655f,
        -0.0008388934f, +0.0017937639f, -0.0034976161f, +0.0063401091f,
        -0.0108412791f, +0.0177062244f, -0.0279626859f, +0.0433318972f,
        -0.0673274067f, +0.1092157212f, -0.2053056334f, +0.7770366035f,
        +0.4722592828f, -0.1718669778f, +0.0964445838f, -0.0604185172f,
        +0.0390138560f, -0.0251015642f, +0.0157850964f, -0.0095689817f,
        +0.0055243410f, -0.0029987796f, +0.0015071050f, -0.0006867310f,
        +0.0002748471f, -0.0000911614f, +0.0000213585f, -0.0000012744f
    },
    {
        -0.0000032671f, +0.0000313486f, -0.0001212397f, +0.0003489973f,
        -0.0008459576f, +0.0018154180f, -0.0035498463f, +0.0064493058f,
        -0.0110478500f, +0.0180685782f, -0.0285616886f, +0.0442757611f,
        -0.0687528444f, +0.1112392555f, -0.2072971213f, +0.7383335558f,
        +0.5191395661f, -0.1825986398f, +0.1017871517f, -0.0636367499f,
        +0.0410775747f, -0.0264424041f, +0.0166455225f, -0.0101054108f,
        +0.0058450504f, -0.0031803656f, +0.0016031094f, -0.0007332736f,
        +0.0002950064f, -0.0000986415f, +0.0000235308f, -0.0000015811f
    },
    {
        -0.0000029453f, +0.0000303707f, -0.0001192314f, +0.0003457548f,
        -0.0008422197f, +0.0018140007f, -0.0035571840f, +0.0064773308f,
        -0.0111159924f, +0.0182054520f, -0.0288055172f, +0.0446705934f,
        -0.0693275455f, +0.1118903228f, -0.2067842857f, +0.6975315552f,
        +0.5654142596f, -0.1916378464f, +0.1060859992f, -0.0661847528f,
        +0.0427057015f, -0.0275034058f, +0.0173310942f, -0.0105369371f,
        +0.0061060692f, -0.0033301778f, +0.0016835662f, -0.0007729959f,
        +0.0003125909f, -0.0001053529f, +0.0000255708f, -0.0000019107f
    },
    {
        -0.0000026035f, +0.0000290450f, -0.0001158103f, +0.0003383623f,
        -0.0008283135f, +0.0017906283f, -0.0035214398f, +0.0064269349f,
        -0.0110496830f, +0.0181224240f, -0.0287019943f, +0.0445279186f,
        -0.0690708056f, +0.1112100673f, -0.2039010961f, +0.6549140203f,
        +0.6107745167f, -0.1987975381f, +0.1092530039f, -0.0680109518f,
        +0.0438652479f, -0.0282628379f, +0.0178275310f, -0.0108543862f,
        +0.0063017213f, -0.0034448798f, +0.0017466391f, -0.0008049685f,
        +0.0003271787f, -0.0001111306f, +0.0000274267f, -0.0000022548f
    },
    {
        -0.0000022548f, +0.0000274267f, -0.0001111306f, +0.0003271787f,
        -0.0008049685f, +0.0017466391f, -0.0034448798f, +0.0063017213f,
        -0.0108543862f, +0.0178275310f, -0.0282628379f, +0.0438652479f,
        -0.0680109518f, +0.1092530039f, -0.1987975381f, +0.6107745167f,
        +0.6549140203f, -0.2039010961f, +0.1112100673f, -0.0690708056f,
        +0.0445279186f, -0.0287019943f, +0.0181224240f, -0.0110496830f,
        +0.0064269349f, -0.0035214398f, +0.0017906283f, -0.0008283135f,
        +0.0003383623f, -0.0001158103f, +0.0000290450f, -0.0000026035f
    },
    {
        -0.0000019107f, +0.0000255708f, -0.0001053529f, +0.0003125909f,
        -0.0007729959f, +0.0016835662f, -0.0033301778f, +0.0061060692f,
        -0.0105369371f, +0.0173310942f, -0.0275034058f, +0.0427057015f,
        -0.0661847528f, +0.1060859992f, -0.1916378464f, +0.5654142596f,
        +0.6975315552f, -0.2067842857f, +0.1118903228f, -0.0693275455f,
        +0.0446705934f, -0.0288055172f, +0.0182054520f, -0.0111159924f,
        +0.0064773308f, -0.0035571840f, +0.0018140007f, -0.0008422197f,
        +0.0003457548f, -0.0001192314f, +0.0000303707f, -0.0000029453f
    },
    {
        -0.0000015811f, +0.0000235308f, -0.0000986415f, +0.0002950064f,
        -0.0007332736f, +0.0016031094f, -0.0031803656f, +0.0058450504f,
        -0.0101054108f, +0.0166455225f, -0.0264424041f, +0.0410775747f,
        -0.0636367499f, +0.1017871517f, -0.1825986398f, +0.5191395661f,
        +0.7383335558f, -0.2072971213f, +0.1112392555f, -0.0687528444f,
        +0.0442757611f, -0.0285616886f, +0.0180685782f, -0.0110478500f,
        +0.0064493058f, -0.0035498463f, +0.0018154180f, -0.0008459576f,
        +0.0003489973f, -0.0001212397f, +0.0000313486f, -0.0000032671f
    },
    {
        -0.0000012744f, +0.0000213585f, -0.0000911614f, +0.0002748471f,
        -0.0006867310f, +0.0015071050f, -0.0029987796f, +0.0055243410f,
        -0.0095689817f, +0.0157850964f, -0.0251015642f, +0.0390138560f,
        -0.0604185172f, +0.0964445838f, -0.1718669778f, +0.4722592828f,
        +0.7770366035f, -0.2053056334f, +0.1092157212f, -0.0673274067f,
        +0.0433318972f, -0.0279626859f, +0.0177062244f, -0.0108412791f,
        +0.0063401091f, -0.0034976161f, +0.0017937639f, -0.0008388934f,
        +0.0003477655f, -0.0001216900f, +0.0000319243f, -0.0000035546f
    },
    {
        -0.0000009968f, +0.0000191031f, -0.0000830761f, +0.0002525421f,
        -0.0006343344f, +0.0013974965f, -0.0027890067f, +0.0051501286f,
        -0.0089377738f, +0.0147657362f, -0.0235052933f, +0.0365517032f,
        -0.0565878630f, +0.0901551619f, -0.1596383586f, +0.4250822109f,
        +0.8133698514f, -0.2006935166f, +0.1057928503f, -0.0650414694f,
        +0.0418337815f, -0.0270047965f, +0.0171154204f, -0.0104938934f,
        +0.0061479103f, -0.0033991812f, +0.0017481703f, -0.0008205037f,
        +0.0003417769f, -0.0001204491f, +0.0000320450f, -0.0000037923f
    },
    {
        -0.0000007529f, +0.0000168103f, -0.0000745454f, +0.0002285221f,
        -0.0005770723f, +0.0012763049f, -0.0025548284f, +0.0047290161f,
        -0.0082227040f, +0.0136047578f, -0.0216803022f, +0.0337318847f,
        -0.0522079819f, +0.0830231596f, -0.1461146806f, +0.3779145552f,
        +0.8470773515f, -0.1933636434f, +0.1009588230f, -0.0618952059f,
        +0.0397827472f, -0.0256885889f, +0.0162959253f, -0.0100049828f,
        +0.0058718582f, -0.0032537663f, +0.0016780403f, -0.0007903882f,
        +0.0003307972f, -0.0001173988f, +0.0000316605f, -0.0000039641f
    },
    {
        -0.0000005454f, +0.0000145225f, -0.0000657233f, +0.0002032129f,
        -0.0005159412f, +0.0011455986f, -0.0023001659f, +0.0042679248f,
        -0.0074353219f, +0.0123206187f, -0.0196552177f, +0.0305981951f,
        -0.0473465686f, +0.0751588798f, -0.1315021858f, +0.3310574185f,
        +0.8779202636f, -0.1832394230f, +0.0947175064f, -0.0578990264f,
        +0.0371868591f, -0.0240190360f, +0.0152503190f, -0.0093755812f,
        +0.0055121300f, -0.0030611663f, +0.0015830695f, -0.0007482824f,
        +0.0003146466f, -0.0001124389f, +0.0000307247f, -0.0000040537f
    },
    {
        -0.0000003753f, +0.0000122775f, -0.0000567561f, +0.0001770302f,
        -0.0004519321f, +0.0010074656f, -0.0020290245f, +0.0037739967f,
        -0.0065876465f, +0.0109326585f, -0.0174601824f, +0.0271968506f,
        -0.0420749059f, +0.0666772507f, -0.1160094089f, +0.2848043664f,
        +0.9056789241f, -0.1702659926f, +0.0870889414f, -0.0530737661f,
        +0.0340610154f, -0.0220055878f, +0.0139840604f, -0.0086085142f,
        +0.0050699679f, -0.0028217731f, +0.0014632641f, -0.0006940681f,
        +0.0002932065f, -0.0001054898f, +0.0000291962f, -0.0000040443f
    },
    {
        -0.0000002417f, +0.0000101091f, -0.0000477810f, +0.0001503745f,
        -0.0003860180f, +0.0008639854f, -0.0017454402f, +0.0032544977f,
        -0.0056920032f, +0.0094608367f, -0.0151264486f, +0.0235758735f,
        -0.0364669358f, +0.0576964130f, -0.0998451493f, +0.2394390841f,
        +0.9301547546f, -0.1544112244f, +0.0781096722f, -0.0474507593f,
        +0.0304269704f, -0.0196621911f, +0.0125055128f, -0.0077084263f,
        +0.0045477043f, -0.0025365958f, +0.0013189554f, -0.0006277834f,
        +0.0002664240f, -0.0000964957f, +0.0000270400f, -0.0000039198f
    },
    {
        -0.0000001425f, +0.0000080461f, -0.0000389244f, +0.0001236264f,
        -0.0003191422f, +0.0007172029f, -0.0014534276f, +0.0027167232f,
        -0.0047608622f, +0.0079254716f, -0.0126859712f, +0.0197844725f,
        -0.0305983271f, +0.0483363121f, -0.0832164873f, +0.1952331456f,
        +0.9511719921f, -0.1356665369f, +0.0678329093f, -0.0410717922f,
        +0.0263132746f, -0.0170072549f, +0.0108259326f, -0.0066817850f,
        +0.0039487735f, -0.0022072742f, +0.0011508111f, -0.0005496306f,
        +0.0002343176f, -0.0000854264f, +0.0000242282f, -0.0000036648f
    },
    {
        -0.0000000740f, +0.0000061130f, -0.0000303015f, +0.0000971433f,
        -0.0002522071f, +0.0005691040f, -0.0011569305f, +0.0021679066f,
        -0.0038066817f, +0.0063469827f, -0.0101710034f, +0.0158724270f,
        -0.0245455488f, +0.0387173115f, -0.0663268640f, +0.1524439164f,
        +0.9685792243f, -0.1140474994f, +0.0563285222f, -0.0339889340f,
        +0.0217551320f, -0.0140635597f, +0.0089594224f, -0.0055368629f,
        +0.0032777097f, -0.0018360835f, +0.0009598425f, -0.0004599821f,
        +0.0001969809f, -0.0000722804f, +0.0000207412f, -0.0000032650f
    },
    {
        -0.0000000316f, +0.0000043292f, -0.0000220148f, +0.0000712550f,
        -0.0001860652f, +0.0004215934f, -0.0008597757f, +0.0016151325f,
        -0.0028417551f, +0.0047456404f, -0.0076137015f, +0.0118894812f,
        -0.0183849603f, +0.0289588411f, -0.0493742406f, +0.1113126084f,
        +0.9822507145f, -0.0895942203f, +0.0436828569f, -0.0262642439f,
        +0.0167941732f, -0.0108581105f, +0.0069228469f, -0.0042836961f,
        +0.0025401311f, -0.0014259322f, +0.0007474069f, -0.0003593845f,
        +0.0001545864f, -0.0000570867f, +0.0000165689f, -0.0000027078f
    },
    {
        -0.0000000094f, +0.0000027095f, -0.0000141538f, +0.0000462618f,
        -0.0001215100f, +0.0002764734f, -0.0005656298f, +0.0010652546f,
        -0.0018780675f, +0.0031413256f, -0.0050457431f, +0.0078847572f,
        -0.0121919281f, +0.0191780947f, -0.0325493562f, +0.0720625023f,
        +0.9920875035f, -0.0623715136f, +0.0299983759f, -0.0179693531f,
        +0.0114781442f, -0.0074219332f, +0.0047357111f, -0.0029340187f,
        +0.0017427091f, -0.0009803506f, +0.0005152066f, -0.0002485605f,
        +0.0001073872f, -0.0000399064f, +0.0000117116f, -0.0000019828f
    },
    {
        -0.0000000012f, +0.0000012645f, -0.0000067946f, +0.0000224320f,
        -0.0000592696f, +0.0001354267f, -0.0002779601f, +0.0005248199f,
        -0.0009271590f, +0.0015533014f, -0.0024979610f, +0.0039061912f,
        -0.0060399793f, +0.0094887898f, -0.0160340981f, +0.0348973545f,
        +0.9980182787f, -0.0324688367f, +0.0153931209f, -0.0091849261f,
        +0.0058605138f, -0.0037898167f, +0.0024200032f, -0.0015011736f,
        +0.0008931230f, -0.0005034712f, +0.0002652818f, -0.0001284086f,
        +0.0000557190f, -0.0000208343f, +0.0000061808f, -0.0000010818f
    },
    {
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
        +1.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
         0.0000000000f,  0.0000000000f,  0.0000000000f,  0.0000000000f
    },
    {
         0.0000000000f, -0.0000010818f, +0.0000061808f, -0.0000208343f,
        +0.0000557190f, -0.0001284086f, +0.0002652818f, -0.0005034712f,
        +0.0008931230f, -0.0015011736f, +0.0024200032f, -0.0037898167f,
        +0.0058605138f, -0.0091849261f, +0.0153931209f, -0.0324688367f,
        +0.9980182787f, +0.0348973545f, -0.0160340981f, +0.0094887898f,
        -0.0060399793f, +0.0039061912f, -0.0024979610f, +0.0015533014f,
        -0.0009271590f, +0.0005248199f, -0.0002779601f, +0.0001354267f,
        -0.0000592696f, +0.0000224320f, -0.0000067946f, +0.0000012645f
    }
};

/* -------------------------------------------------------------------------- */
