package kr.aerial.observer

import android.graphics.Bitmap
import android.graphics.RectF
import java.io.Closeable
import kotlin.math.*

/** JNI adapter for the same pinned official ByteTrack C++ sources used by the iOS app. */
class AerialTracker : Closeable {
    companion object { init { System.loadLibrary("aerial_tracker") } }
    private external fun create(): Long
    private external fun destroy(handle: Long)
    private external fun updateNative(handle: Long, rows: FloatArray): FloatArray?
    private var handle = create()
    private data class Evidence(var rect: RectF, var last: Long, var scores: FloatArray? = null,
                                var hits: Int = 0, var decision: String? = null, var since: Long = 0)
    private val evidence = mutableMapOf<Int, Evidence>()
    private var lastTime = 0L
    private var offsetX = 0f
    private var offsetY = 0f
    val foreground get() = evidence.values.map { it.rect }
    fun reset() {
        destroy(handle); handle = create(); evidence.clear(); offsetX = 0f; offsetY = 0f; lastTime = 0
    }
    fun update(boxes: List<Detection>, time: Long, motion: CameraMotion.Estimate): List<Observation> {
        val dt = if (lastTime == 0L) 50L else time-lastTime
        if (dt < 0 || dt > 1000) reset()
        lastTime = time
        offsetX += motion.dx; offsetY += motion.dy
        val valid = boxes.withIndex().filter { (_, b) -> b.score >= .10f && b.rect.width() in 0f..1f && b.rect.height() in 0f..1f }.take(50)
        val input = valid.flatMap { (index, b) -> listOf((b.rect.left-offsetX)*640,
            (b.rect.top-offsetY)*640, b.rect.width()*640, b.rect.height()*640, b.score, index.toFloat()) }.toFloatArray()
        val native = updateNative(handle, input) ?: error("ByteTrack input/engine failure")
        if (motion.disrupted) evidence.values.forEach { it.scores=null; it.hits=0; it.decision=null; it.since=time }
        evidence.entries.removeAll { time-it.value.last > 1000 }
        val output = ArrayList<Observation>()
        val observed = mutableSetOf<Int>()
        val assigned = mutableSetOf<Int>()
        for (i in native.indices step 7) {
            val id = native[i+5].toInt(); val index = native[i+6].toInt()
            if (index !in boxes.indices) continue
            val b = boxes[index]
            val rect = RectF(native[i]/640+offsetX, native[i+1]/640+offsetY,
                (native[i]+native[i+2])/640+offsetX, (native[i+1]+native[i+3])/640+offsetY)
            val current = Decision.label(b.cls, b.score, b.scores)
            val decision = when(current) { "드론", "드론 의심" -> "drone"; "새", "새 의심" -> "bird"; else -> null }
            val e = evidence.getOrPut(id) { Evidence(rect,time,since=time) }
            if (time-e.last > 350 || decision == null || decision != e.decision) {
                e.scores=null; e.hits=0; e.since=time
            }
            e.rect=rect; e.last=time; e.decision=decision
            if (b.score >= .30f && !motion.disrupted) {
                e.hits++
                val alpha = (1-exp(-dt.coerceIn(10,350)/200.0)).toFloat()
                e.scores = e.scores?.mapIndexed { j,v -> v+alpha*(b.scores[j]-v) }?.toFloatArray() ?: b.scores.copyOf()
            }
            var label = Decision.UNKNOWN
            val scores = e.scores
            if (!motion.disrupted && b.score >= .30f && e.hits >= 3 && scores != null) {
                val winner = if (scores[0] >= scores[1]) 0 else 1
                val accumulated = Decision.label(winner,scores[winner],scores)
                if (accumulated == current && (accumulated !in listOf("드론","새") || time-e.since >= 250)) label=accumulated
            }
            observed.add(id); assigned.add(index)
            output.add(Observation(rect,label,b.score,id))
        }
        evidence.forEach { (id,e) ->
            if (id !in observed && !motion.disrupted && time-e.last <= 250)
                output.add(Observation(RectF(e.rect),"관측 끊김",0f,id,false))
        }
        valid.filter { it.index !in assigned && (it.value.score >= .30f || it.value.firstSighting) }.forEach { (_,b) ->
            var label = Decision.label(b.cls,b.score,b.scores)
            if (label in listOf("드론","새")) label += " 의심"
            if (motion.disrupted || b.firstSighting) label=Decision.UNKNOWN
            output.add(Observation(b.rect,label,b.score))
        }
        return output
    }
    override fun close() { destroy(handle) }
}

/** Bounded sparse optical flow; source is the already prepared inference bitmap, no extra full-frame copy. */
class CameraMotion {
    data class Estimate(val dx: Float=0f, val dy: Float=0f, val status: String="보정 OFF", val disrupted: Boolean=false)
    private var previous: IntArray? = null
    private var lastTime=0L
    private var lastZoom=0f
    private var movingUntil=0L
    fun reset() { previous=null; lastTime=0 }
    fun estimate(bitmap: Bitmap, content: RectF, time: Long, zoom: Float, gyro: Float, foreground: List<RectF>): Estimate {
        val w=96; val h=64
        val current = IntArray(w*h)
        for (y in 0 until h) for (x in 0 until w) {
            val c=bitmap.getPixel((content.left+x*content.width()/w).toInt().coerceIn(0,639),
                (content.top+y*content.height()/h).toInt().coerceIn(0,639))
            current[y*w+x]=(((c shr 16) and 255)*77+((c shr 8) and 255)*150+(c and 255)*29) shr 8
        }
        val old=previous; val dt=time-lastTime
        val changed=lastZoom > 0 && abs(zoom/lastZoom-1) > .025
        previous=current; lastTime=time; lastZoom=zoom
        if (changed || gyro > 1.5) return Estimate(status="줌/급이동 재평가",disrupted=true)
        if (old==null || dt <= 0 || dt >= 350) return Estimate(status="보정 기준 준비",disrupted=old!=null)
        fun excluded(x: Int,y: Int) = foreground.any { x/w.toFloat() in (it.left-.04f)..(it.right+.04f) && y/h.toFloat() in (it.top-.04f)..(it.bottom+.04f) }
        var diff=0; var count=0
        for (y in 4 until h-4 step 4) for(x in 4 until w-4 step 4) if (!excluded(x,y)) { diff+=abs(current[y*w+x]-old[y*w+x]); count++ }
        if (gyro > .06 || count > 0 && diff/count > 7) movingUntil=time+250
        if (time >= movingUntil) return Estimate(status="고정 · 보정 생략")
        val deadline=System.nanoTime()+8_000_000
        val flow=mutableListOf<Pair<Int,Int>>()
        for (y in 8 until h-8 step 8) for (x in 8 until w-8 step 8) {
            if (System.nanoTime()>deadline) return Estimate(status="보정 시간 제한 · 기본 추적")
            if (excluded(x,y)) continue
            var minimum=255; var maximum=0
            for(j in -2..2) for(i in -2..2) { val v=old[(y+j)*w+x+i]; minimum=min(minimum,v); maximum=max(maximum,v) }
            if (maximum-minimum <= 24) continue
            var best=Int.MAX_VALUE; var second=Int.MAX_VALUE; var bx=0; var by=0
            for(dy in -4..4) for(dx in -4..4) {
                var error=0
                for(j in -2..2) for(i in -2..2) error+=abs(old[(y+j)*w+x+i]-current[(y+j+dy)*w+x+i+dx])
                if(error<best) { second=best; best=error; bx=dx; by=dy } else second=min(second,error)
            }
            if (best<350 && (best+1).toFloat()/(second+1)<.8f) flow.add(bx to by)
        }
        if (flow.size<8) return Estimate(status="보정 불확실 · 기본 추적")
        val dx=flow.map{it.first}.sorted()[flow.size/2]; val dy=flow.map{it.second}.sorted()[flow.size/2]
        val agreement=flow.count{abs(it.first-dx)<=1 && abs(it.second-dy)<=1}.toFloat()/flow.size
        if(agreement<.65 || abs(dx)>=4 || abs(dy)>=4) return Estimate(status="보정 불확실 · 기본 추적")
        return Estimate(dx/w.toFloat(),dy/h.toFloat(),"이동 · 보정 적용")
    }
}
