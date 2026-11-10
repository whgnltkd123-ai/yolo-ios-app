package kr.aerial.observer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.*
import androidx.camera.core.ImageProxy
import org.json.JSONObject
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.*

data class Detection(val rect: RectF, val score: Float, val cls: Int, val scores: FloatArray,
                     var firstSighting: Boolean = false)
data class Observation(val rect: RectF, val label: String, val score: Float,
                       val id: Int = 0, val observed: Boolean = true)

object Decision {
    const val UNKNOWN = "미확인 물체"
    fun label(cls: Int, score: Float, values: FloatArray): String {
        if (cls !in 0..2 || !score.isFinite() || score !in 0f..1f || values.size != 3 ||
            values.any { !it.isFinite() || it !in 0f..1f }) return UNKNOWN
        if (cls == 2) return if (score >= .60f) "비행기" else UNKNOWN
        val selected = values[cls]
        val competitor = values[1 - cls]
        if (abs(selected - score) >= .02f || selected < .30f || competitor >= .30f ||
            selected - competitor <= .200001f || values[2] >= .30f ||
            selected - values[2] <= .200001f) return UNKNOWN
        return (if (cls == 0) "드론" else "새") + if (selected >= .60f) "" else " 의심"
    }
    fun level(label: String): Int = when (label) {
        "드론" -> 3
        "드론 의심" -> 2
        UNKNOWN -> 1
        else -> 0
    }
}

class AerialDetector(context: Context) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(2)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        addXnnpack(mapOf("intra_op_num_threads" to "2"))
    }
    private val session: OrtSession
    val canvasBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var source: Bitmap? = null
    private var paddedBytes: ByteBuffer? = null
    private val pixels = IntArray(640 * 640)
    private val floats = ByteBuffer.allocateDirect(3 * 640 * 640 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val input = OnnxTensor.createTensor(environment, floats, longArrayOf(1, 3, 640, 640))
    var content = RectF(0f, 0f, 640f, 640f); private set
    private var previous = emptyList<RectF>()
    private var previousTime = 0L
    private var previousZoom = 0f
    private var previousGeometry = ""

    init {
        val metadata = JSONObject(context.assets.open("model-metadata.json").bufferedReader().use { it.readText() })
        check(metadata.getBoolean("conversion_passed")) { "모델 변환 검증이 필요합니다" }
        val expected = metadata.getString("onnx_sha256")
        val file = java.io.File(context.noBackupFilesDir, "aerial-$expected.onnx")
        fun digest(file: java.io.File): String {
            val sha = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(1024 * 1024)
                while (true) { val n = stream.read(buffer); if (n < 0) break; sha.update(buffer, 0, n) }
            }
            return sha.digest().joinToString("") { "%02x".format(it) }
        }
        if (!file.exists() || digest(file) != expected) {
            context.assets.open("aerial.onnx").use { src -> file.outputStream().use { src.copyTo(it) } }
        }
        check(digest(file) == expected) { "모델 파일 무결성 오류" }
        session = environment.createSession(file.path, options)
    }

    // CameraX owns the full-resolution preview. Only the analysis queue touches these buffers.
    fun analyze(image: ImageProxy, time: Long, zoom: Float): List<Detection> {
        val plane = image.planes[0]
        check(plane.pixelStride == 4) { "RGBA camera output required" }
        val paddedWidth = plane.rowStride / 4
        if (source?.width != paddedWidth || source?.height != image.height) {
            source?.recycle()
            source = Bitmap.createBitmap(paddedWidth, image.height, Bitmap.Config.ARGB_8888)
        }
        val bytes = plane.buffer
        bytes.rewind()
        // CameraX RGBA plane contains R,G,B,A, compatible with native ARGB_8888 byte layout.
        val required=plane.rowStride*image.height
        if(bytes.remaining()>=required) source!!.copyPixelsFromBuffer(bytes)
        else {
            // Camera buffers may omit the final row's unused stride padding.
            if(paddedBytes?.capacity()!=required)paddedBytes=ByteBuffer.allocateDirect(required)
            val complete=paddedBytes!!; complete.clear(); complete.put(bytes)
            while(complete.hasRemaining())complete.put(0)
            complete.rewind(); source!!.copyPixelsFromBuffer(complete)
        }
        val crop = RectF(image.cropRect)
        val rotation = image.imageInfo.rotationDegrees
        val rotated = RectF(0f, 0f, crop.width(), crop.height())
        val rotate = Matrix().apply { setRotate(rotation.toFloat()); mapRect(rotated) }
        val gain = 640f / max(rotated.width(), rotated.height())
        val padX = ((640 - (rotated.width() * gain).roundToInt()) / 2f - .1f).roundToInt().toFloat()
        val padY = ((640 - (rotated.height() * gain).roundToInt()) / 2f - .1f).roundToInt().toFloat()
        val transform = Matrix().apply {
            setTranslate(-crop.left, -crop.top)
            postConcat(rotate)
            postTranslate(-rotated.left, -rotated.top)
            postScale(gain, gain)
            postTranslate(padX, padY)
        }
        content = RectF(padX, padY, padX + rotated.width()*gain, padY + rotated.height()*gain)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.save(); canvas.clipRect(content); canvas.drawBitmap(source!!, transform, paint); canvas.restore()
        canvasBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        val n = pixels.size
        for (i in pixels.indices) {
            val value = pixels[i]
            floats.put(i, ((value shr 16) and 255) / 255f)
            floats.put(n+i, ((value shr 8) and 255) / 255f)
            floats.put(2*n+i, (value and 255) / 255f)
        }
        val detections = ArrayList<Detection>()
        session.run(mapOf("images" to input)).use { result ->
            val output = (result[0] as OnnxTensor).floatBuffer
            while (output.remaining() >= 9) {
                val row = FloatArray(9); output.get(row)
                if (row.any { !it.isFinite() } || row[4] <= .10f || row[5].toInt() !in 0..2) continue
                val rect = RectF((row[0]-content.left)/content.width(), (row[1]-content.top)/content.height(),
                    (row[2]-content.left)/content.width(), (row[3]-content.top)/content.height())
                if(!rect.intersect(0f, 0f, 1f, 1f)) continue
                if (rect.width() <= 0 || rect.height() <= 0) continue
                detections.add(Detection(rect, row[4], row[5].toInt(), row.copyOfRange(6, 9)))
                if (detections.size == 50) break
            }
        }
        val geometry = "${image.width}:${image.height}:$rotation:$crop"
        val weak = detections.firstOrNull { b -> b.cls < 2 && b.score >= .20f && b.score < .30f &&
            min(b.rect.width()*content.width(), b.rect.height()*content.height()) <= 32 &&
            detections.none { it.score >= .30f && overlap(it.rect, b.rect) >= .25f } }
        if (time > previousTime && time - previousTime <= 250 && zoom == previousZoom && geometry == previousGeometry &&
            weak != null && previous.any { overlap(weak.rect, it) >= .30f }) weak.firstSighting = true
        previous = detections.filter { it.cls < 2 && it.score >= .20f }.map { RectF(it.rect) }
        previousTime = time; previousZoom = zoom; previousGeometry = geometry
        return detections
    }

    override fun close() {
        input.close(); session.close(); options.close(); source?.recycle(); canvasBitmap.recycle()
    }
}

fun overlap(a: RectF, b: RectF): Float {
    val common = max(0f, min(a.right,b.right)-max(a.left,b.left)) *
        max(0f, min(a.bottom,b.bottom)-max(a.top,b.top))
    return common / max(1e-10f, a.width()*a.height()+b.width()*b.height()-common)
}
