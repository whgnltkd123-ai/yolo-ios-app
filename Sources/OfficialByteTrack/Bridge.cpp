#include "OfficialByteTrack.h"
#include "BYTETracker.h"
#include <algorithm>
#include <cmath>
void *obt_create(void) { return new BYTETracker(30, 30); }
void obt_destroy(void *handle) { delete static_cast<BYTETracker *>(handle); }
int32_t obt_update(void *handle, const OBTDetection *input, int32_t count,
                  OBTTrack *output, int32_t capacity) {
    if (!handle || count < 0 || count > 50 || capacity < 0 || (count && !input) || !output) return -1;
    try {
    std::vector<Object> objects;
    objects.reserve(count);
    for (int i=0; i<count; ++i) {
        const auto &b = input[i];
        if (!std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.width)
            || !std::isfinite(b.height) || !std::isfinite(b.score)
            || b.width<=0 || b.height<=0 || b.score<0 || b.score>1) return -1;
        objects.push_back({cv::Rect_<float>(b.x,b.y,b.width,b.height),b.index,b.score});
    }
    const auto tracks = static_cast<BYTETracker *>(handle)->update(objects);
    const auto n = std::min(static_cast<int>(tracks.size()), static_cast<int>(capacity));
    for (int i=0;i<n;++i) {
        const auto &t=tracks[i];
        output[i]={t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3],t.score,t.track_id,t.detection_index};
    }
    return n;
    } catch (...) { return -2; }
}
