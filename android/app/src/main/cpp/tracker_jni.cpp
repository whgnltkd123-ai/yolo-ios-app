#include <jni.h>
#include "OfficialByteTrack.h"
#include <vector>

extern "C" JNIEXPORT jlong JNICALL
Java_kr_aerial_observer_AerialTracker_create(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(obt_create());
}
extern "C" JNIEXPORT void JNICALL
Java_kr_aerial_observer_AerialTracker_destroy(JNIEnv *, jobject, jlong handle) {
    obt_destroy(reinterpret_cast<void *>(handle));
}
extern "C" JNIEXPORT jfloatArray JNICALL
Java_kr_aerial_observer_AerialTracker_updateNative(JNIEnv *env, jobject, jlong handle, jfloatArray rows) {
    const int length = env->GetArrayLength(rows);
    if (length % 6 || length > 300) return nullptr;
    std::vector<float> values(length);
    env->GetFloatArrayRegion(rows, 0, length, values.data());
    std::vector<OBTDetection> input;
    for (int i = 0; i < length; i += 6)
        input.push_back({values[i],values[i+1],values[i+2],values[i+3],values[i+4],static_cast<int>(values[i+5])});
    OBTTrack tracks[50];
    const auto n = obt_update(reinterpret_cast<void *>(handle), input.data(), input.size(), tracks, 50);
    if (n < 0) return nullptr;
    std::vector<float> output;
    for (int i = 0; i < n; ++i) {
        const auto &t = tracks[i];
        output.insert(output.end(), {t.x,t.y,t.width,t.height,t.score,static_cast<float>(t.id),static_cast<float>(t.index)});
    }
    auto result = env->NewFloatArray(output.size());
    env->SetFloatArrayRegion(result, 0, output.size(), output.data());
    return result;
}
