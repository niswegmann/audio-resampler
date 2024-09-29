#include "drb-audio-converter.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum { source_sampling_rate = drb_audio_converter_sampling_rate_44100 };
enum { target_sampling_rate = drb_audio_converter_sampling_rate_48000 };
enum { channel_count = 2 };
enum { block_size = drb_audio_converter_block_size_4 };
enum { max_block_count = 64 };
enum { total_frame_count = 500 };
enum { quality = drb_audio_converter_quality_good };

static int const slices [9] = { 13, 17, 4, 7, 5, 4, 21, 29, 300 };

enum { slice_count = sizeof(slices) / sizeof(int) };

static void consume_frames
    (
        void * user_data,
        double timestamp,
        DrB_Audio_Converter_Buffer const buffers [],
        int frame_count
    )
{
    (void)user_data, (void)timestamp;

    static int count = 0;

    assert(frame_count % block_size == 0);

    for (int frame = 0; frame < frame_count; frame++)
    {
        printf("%3d", count++);

        for (int channel = 0; channel < channel_count; channel++)
        {
            printf(", %8.3f", buffers[channel].samples[frame]);
        }

        printf("\n");
    }
}

extern int main (int const argc, char const * const argv [const])
{
    (void)argc, (void)argv;

    long alignment, size;

    assert(drb_audio_converter_alignment_and_size
    (
        source_sampling_rate,
        target_sampling_rate,
        channel_count,
        block_size,
        max_block_count,
        drb_audio_converter_direction_push,
        quality,
        &alignment,
        &size
    ));

    void * const converter_memory = aligned_alloc(alignment, size);

    assert(converter_memory);

    DrB_Audio_Converter * const converter = drb_audio_converter_construct
    (
        converter_memory,
        source_sampling_rate,
        target_sampling_rate,
        channel_count,
        block_size,
        max_block_count,
        drb_audio_converter_direction_push,
        quality,
        (DrB_Audio_Converter_Data_Callback){ .process = consume_frames }
    );

    assert(converter);

    drb_audio_converter_work_memory_alignment_and_size
    (
        converter,
        &alignment,
        &size
    );

    void * const work_memory = aligned_alloc(alignment, size);

    assert(work_memory);

    float samples [channel_count][total_frame_count];

    for (int channel = 0; channel < channel_count; channel++)
    {
        for (int frame = 0; frame < total_frame_count; frame++)
        {
            samples[channel][frame] = frame + 100 * channel;
        }
    }

    for (int index = 0, offset = 0; index < slice_count; index++)
    {
        DrB_Audio_Converter_Buffer buffers [channel_count];

        for (int channel = 0; channel < channel_count; channel++)
        {
            buffers[channel].samples = samples[channel] + offset;
        }

        drb_audio_converter_process
        (
            converter,
            work_memory,
            0.0,
            buffers,
            slices[index]
        );

        offset += slices[index];

        assert(offset <= total_frame_count);
    }

    free(converter);

    return EXIT_SUCCESS;
}
