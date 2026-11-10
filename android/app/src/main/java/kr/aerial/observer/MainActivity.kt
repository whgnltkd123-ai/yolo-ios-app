package kr.aerial.observer

import android.Manifest
import android.app.AlertDialog
import android.content.*
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.*
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.*
import android.provider.MediaStore
import android.view.*
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.*

class MainActivity : ComponentActivity(), SensorEventListener {
    private val analysisQueue = Executors.newSingleThreadExecutor()
    private val fileQueue = Executors.newSingleThreadExecutor()
    private var detector: AerialDetector? = null
    private var tracker: AerialTracker? = null
    private val motion = CameraMotion()
    private lateinit var preview: PreviewView
    private lateinit var overlay: AerialOverlay
    private lateinit var status: TextView
    private lateinit var alarm: AerialAlarm
    private lateinit var sensors: SensorManager
    private var camera: Camera? = null
    private var provider: ProcessCameraProvider? = null
    @Volatile private var resumed=false
    @Volatile private var mode=0
    @Volatile private var zoom=1f
    @Volatile private var gyro=0f
    @Volatile private var evidenceEnabled=false
    private var activeMode=0
    private var lastEvidence=0L
    private var lastStatus=0L
    private var previousAnalysis=0L
    private var evidenceExport: File? = null
    private val evidenceDir get() = File(filesDir,"evidence").apply { mkdirs() }
    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text="카메라 권한을 허용해야 관측할 수 있습니다. 앱을 다시 열어 권한을 허용해 주세요."
    }
    private val saveDocument = registerForActivityResult(ActivityResultContracts.CreateDocument("image/jpeg")) { uri ->
        val file=evidenceExport; evidenceExport=null
        if(uri!=null && file!=null) fileQueue.execute {
            runCatching { contentResolver.openOutputStream(uri)!!.use { dest -> file.inputStream().use{it.copyTo(dest)} } }
                .onSuccess { runOnUiThread { toast("선택한 위치에 저장했습니다") } }
                .onFailure { showError("저장 실패", it) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        alarm=AerialAlarm(this)
        sensors=getSystemService(SENSOR_SERVICE) as SensorManager
        val root=LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setBackgroundColor(Color.rgb(16,24,32)) }
        ViewCompat.setOnApplyWindowInsetsListener(root) { view,insets ->
            val bars=insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left,bars.top,bars.right,bars.bottom); insets
        }
        status=TextView(this).apply { text="모델 준비 중…"; textSize=12f; setTextColor(Color.WHITE); setPadding(12,10,12,10) }
        root.addView(status)
        val modes=Spinner(this).apply {
            adapter=ArrayAdapter(this@MainActivity,android.R.layout.simple_spinner_dropdown_item,
                arrayOf("1 · 기존 AI 모델", "2 · 객체 ID 추적", "3 · 움직임 보정 추적"))
            onItemSelectedListener=object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?,view: View?,position: Int,id: Long) { mode=position }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
        root.addView(modes)
        val cameraPanel=FrameLayout(this)
        preview=PreviewView(this).apply { implementationMode=PreviewView.ImplementationMode.PERFORMANCE; scaleType=PreviewView.ScaleType.FIT_CENTER }
        overlay=AerialOverlay(this)
        cameraPanel.addView(preview,FrameLayout.LayoutParams(-1,-1))
        cameraPanel.addView(overlay,FrameLayout.LayoutParams(-1,-1))
        val credit=TextView(this).apply { text="Made by 중령 김옥전, 소령 최현민, 대위 조휘상"; textSize=8f; setTextColor(0x99FFFFFF.toInt()); setPadding(4,2,4,2) }
        cameraPanel.addView(credit,FrameLayout.LayoutParams(-2,-2,Gravity.BOTTOM or Gravity.END))
        root.addView(cameraPanel,LinearLayout.LayoutParams(-1,0,1f))
        val zoomLabel=TextView(this).apply { text="줌 1.0× · 직접 확대"; setTextColor(Color.WHITE); textSize=12f }
        val seek=SeekBar(this).apply { max=90 }
        seek.setOnSeekBarChangeListener(object:SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar:SeekBar?,value:Int,fromUser:Boolean) {
                if(!fromUser)return
                val state=camera?.cameraInfo?.zoomState?.value ?: return
                zoom=(1f+value/10f).coerceIn(state.minZoomRatio,state.maxZoomRatio)
                camera?.cameraControl?.setZoomRatio(zoom)
                zoomLabel.text="줌 %.1f× · 기기 최대 %.1f×".format(zoom,state.maxZoomRatio)
            }
            override fun onStartTrackingTouch(bar:SeekBar?) {}
            override fun onStopTrackingTouch(bar:SeekBar?) {}
        })
        val scale=ScaleGestureDetector(this,object:ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d:ScaleGestureDetector):Boolean {
                val state=camera?.cameraInfo?.zoomState?.value ?: return false
                zoom=(zoom*d.scaleFactor).coerceIn(max(1f,state.minZoomRatio),min(10f,state.maxZoomRatio))
                camera?.cameraControl?.setZoomRatio(zoom); seek.progress=((zoom-1)*10).roundToInt()
                zoomLabel.text="줌 %.1f× · 직접 확대".format(zoom); return true
            }
        })
        preview.setOnTouchListener { view,event ->
            scale.onTouchEvent(event)
            if(event.actionMasked==MotionEvent.ACTION_UP)view.performClick()
            true
        }
        root.addView(zoomLabel); root.addView(seek)
        val switches=LinearLayout(this)
        switches.addView(Switch(this).apply { text="경보"; isChecked=true; setOnCheckedChangeListener { _,on -> alarm.enabled=on; if(!on)alarm.stop() } })
        switches.addView(Switch(this).apply { text="자동 채증"; isChecked=false; setOnCheckedChangeListener { _,on -> evidenceEnabled=on } })
        root.addView(switches)
        val buttons=LinearLayout(this)
        buttons.addView(Button(this).apply { text="갤러리"; setOnClickListener { openGallery() } },LinearLayout.LayoutParams(0,-2,1f))
        buttons.addView(Button(this).apply { text="안내 / 라이선스"; setOnClickListener { showInformation() } },LinearLayout.LayoutParams(0,-2,1f))
        root.addView(buttons); setContentView(root)
        analysisQueue.execute {
            try {
                detector=AerialDetector(this)
                runOnUiThread {
                    status.text="준비 완료 · 카메라 영상은 기기 안에서만 처리"
                    if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED) startCamera()
                    else AlertDialog.Builder(this).setTitle("카메라 사용 안내")
                        .setMessage("드론·새 관측에 카메라를 사용합니다. 영상은 서버로 전송하지 않습니다. 자동 채증을 켜면 선택한 관측 장면을 앱 내 갤러리에 저장합니다.")
                        .setPositiveButton("계속") { _,_ -> requestCamera.launch(Manifest.permission.CAMERA) }
                        .setNegativeButton("취소",null).show()
                }
            } catch(e:Exception) { showError("모델을 준비하지 못했습니다",e) }
        }
    }

    private fun startCamera() {
        preview.post {
            if(preview.width==0 || preview.height==0 || isDestroyed)return@post
            val future=ProcessCameraProvider.getInstance(this)
            future.addListener({
                try {
                    provider=future.get()
                    val view=Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
                    val analysis=ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
                    analysis.setAnalyzer(analysisQueue) { frame ->
                        try { if(resumed) analyze(frame) } catch(e:Exception) {
                            analysis.clearAnalyzer(); runOnUiThread { alarm.stop(); overlay.objects=emptyList(); overlay.invalidate() }
                            showError("영상 분석 오류",e)
                        } finally { frame.close() }
                    }
                    val group=UseCaseGroup.Builder().addUseCase(view).addUseCase(analysis)
                        .setViewPort(preview.viewPort!!).build()
                    provider!!.unbindAll()
                    camera=provider!!.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group)
                } catch(e:Exception) { showError("카메라 시작 오류",e) }
            },ContextCompat.getMainExecutor(this))
        }
    }
    private fun analyze(frame:ImageProxy) {
        val engine=detector ?: return
        val start=SystemClock.elapsedRealtime()
        val boxes=engine.analyze(frame,start,zoom)
        val currentMode=mode
        if(currentMode!=activeMode) { tracker?.close(); tracker=null; motion.reset(); activeMode=currentMode }
        val estimate=if(currentMode==2) motion.estimate(engine.canvasBitmap,engine.content,start,zoom,gyro,tracker?.foreground ?: emptyList()) else CameraMotion.Estimate()
        val objects=if(currentMode==0) boxes.filter { it.score>=.30f || it.firstSighting }.map {
            Observation(it.rect,if(it.firstSighting)Decision.UNKNOWN else Decision.label(it.cls,it.score,it.scores),it.score)
        } else {
            if(tracker==null)tracker=AerialTracker()
            tracker!!.update(boxes,start,estimate)
        }
        val elapsed=SystemClock.elapsedRealtime()-start
        val fps=if(previousAnalysis==0L)0.0 else 1000.0/max(1,start-previousAnalysis)
        previousAnalysis=start
        val aspect=engine.content.width()/engine.content.height()
        if(evidenceEnabled && start-lastEvidence>=5000 && objects.any{it.observed && it.score>=.30f}) {
            lastEvidence=start
            val bitmap=Bitmap.createBitmap(engine.canvasBitmap,engine.content.left.roundToInt(),engine.content.top.roundToInt(),
                engine.content.width().roundToInt().coerceAtMost(640),engine.content.height().roundToInt().coerceAtMost(640))
                .copy(Bitmap.Config.ARGB_8888,true)
            val captured=System.currentTimeMillis()
            fileQueue.execute { saveEvidence(bitmap,objects,captured) }
        }
        val message=if(start-lastStatus>=500) {
            lastStatus=start
            "AI %.1f FPS · 분석 %d ms · %s".format(fps,elapsed,if(currentMode==0)"추적 OFF" else estimate.status)
        } else null
        runOnUiThread {
            if(!resumed)return@runOnUiThread
            overlay.aspect=aspect; overlay.objects=objects; overlay.invalidate()
            alarm.observe(objects.filter{it.observed && it.score>=.30f}.maxOfOrNull{Decision.level(it.label)} ?: 0)
            if(message!=null)status.text=message
        }
    }

    private fun saveEvidence(bitmap:Bitmap,objects:List<Observation>,time:Long) {
        try {
            val c=Canvas(bitmap); val p=Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.YELLOW; style=Paint.Style.STROKE; strokeWidth=2f }
            val list=JSONArray()
            objects.filter{it.observed}.forEach {
                val r=it.rect
                c.drawRect(r.left*bitmap.width,r.top*bitmap.height,r.right*bitmap.width,r.bottom*bitmap.height,p)
                list.put(JSONObject().put("label",it.label).put("score",it.score.toDouble()).put("id",it.id)
                    .put("rect",JSONArray(listOf(r.left,r.top,r.right,r.bottom))))
            }
            File(evidenceDir,"$time.jpg").outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG,92,it) }
            File(evidenceDir,"$time.json").writeText(JSONObject().put("capturedAt",time).put("objects",list)
                .put("source","same inference frame; 640-pixel model input crop").toString(2))
        } catch(e:Exception) { showError("채증 저장 실패",e) } finally { bitmap.recycle() }
    }
    private fun openGallery() {
        fileQueue.execute {
            val files=evidenceDir.listFiles{f->f.extension=="jpg"}?.sortedByDescending{it.name}.orEmpty()
            runOnUiThread {
                if(files.isEmpty()) { toast("저장된 채증이 없습니다"); return@runOnUiThread }
                AlertDialog.Builder(this).setTitle("앱 내 갤러리 · ${files.size}장")
                    .setItems(files.map{java.text.SimpleDateFormat("MM-dd HH:mm:ss",java.util.Locale.KOREA).format(java.util.Date(it.nameWithoutExtension.toLong()))}.toTypedArray()) { _,i -> showEvidence(files[i]) }
                    .setNegativeButton("닫기",null).show()
            }
        }
    }
    private fun showEvidence(file:File) {
        val image=ImageView(this).apply { adjustViewBounds=true; setImageURI(android.net.Uri.fromFile(file)) }
        AlertDialog.Builder(this).setTitle("채증 이미지").setView(image)
            .setPositiveButton("저장/공유") { _,_ -> AlertDialog.Builder(this).setItems(arrayOf("사진 저장","다른 앱으로 공유")) { _,i ->
                if(i==0)exportEvidence(file) else {
                    val uri=FileProvider.getUriForFile(this,"$packageName.files",file)
                    startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("image/jpeg")
                        .putExtra(Intent.EXTRA_STREAM,uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),"채증 공유"))
                }
            }.show() }
            .setNeutralButton("삭제") { _,_ -> AlertDialog.Builder(this).setMessage("이 채증 이미지와 판정 기록을 삭제할까요?")
                .setPositiveButton("삭제") { _,_ -> fileQueue.execute { file.delete(); File(file.parentFile,file.nameWithoutExtension+".json").delete() } }
                .setNegativeButton("취소",null).show() }
            .setNegativeButton("닫기",null).show()
    }
    private fun exportEvidence(file:File) {
        if(Build.VERSION.SDK_INT<29) { evidenceExport=file; saveDocument.launch("Aerial-${file.name}"); return }
        fileQueue.execute {
            var uri:android.net.Uri?=null
            try {
                val values=ContentValues().apply { put(MediaStore.Images.Media.DISPLAY_NAME,"Aerial-${file.name}")
                    put(MediaStore.Images.Media.MIME_TYPE,"image/jpeg"); put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/AerialObserver")
                    put(MediaStore.Images.Media.IS_PENDING,1) }
                uri=contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values) ?: error("사진 저장 위치 없음")
                contentResolver.openOutputStream(uri)!!.use{dest->file.inputStream().use{it.copyTo(dest)}}
                contentResolver.update(uri,ContentValues().apply{put(MediaStore.Images.Media.IS_PENDING,0)},null,null)
                runOnUiThread{toast("사진 갤러리에 저장했습니다")}
            } catch(e:Exception) { uri?.let{contentResolver.delete(it,null,null)}; showError("사진 저장 실패",e) }
        }
    }
    private fun showInformation() {
        fun showText(path:String) {
            val text=assets.open(path).bufferedReader().use{it.readText()}
            val view=TextView(this).apply { this.text=text; textSize=14f; setPadding(20,20,20,20) }
            AlertDialog.Builder(this).setView(ScrollView(this).apply{addView(view)}).setPositiveButton("닫기",null).show()
        }
        AlertDialog.Builder(this).setTitle("공중객체 관측").setItems(arrayOf("판정과 사용 안내","개인정보처리방침","오픈소스 및 모델 출처","라이선스 전문")) { _,index ->
            if(index<3)showText(arrayOf("usage.txt","privacy.txt","notices.txt")[index])
            else {
                val names=assets.list("licenses")!!.sorted().toTypedArray()
                AlertDialog.Builder(this).setTitle("라이선스 전문").setItems(names) { _,i -> showText("licenses/${names[i]}") }.show()
            }
        }.show()
    }
    private fun toast(text:String) { Toast.makeText(this,text,Toast.LENGTH_SHORT).show() }
    private fun showError(message:String,e:Throwable) { runOnUiThread { if(!isDestroyed) { status.text="$message: ${e.message}"; toast(message) } } }
    override fun onResume() { super.onResume(); resumed=true; sensors.registerListener(this,sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE),SensorManager.SENSOR_DELAY_GAME) }
    override fun onPause() { resumed=false; sensors.unregisterListener(this); alarm.stop(); overlay.objects=emptyList(); overlay.invalidate(); super.onPause() }
    override fun onDestroy() {
        provider?.unbindAll(); alarm.close()
        analysisQueue.execute { tracker?.close(); detector?.close() }; analysisQueue.shutdown(); fileQueue.shutdown()
        super.onDestroy()
    }
    override fun onSensorChanged(event:SensorEvent) { gyro=sqrt(event.values.take(3).sumOf{(it*it).toDouble()}).toFloat() }
    override fun onAccuracyChanged(sensor:Sensor?,accuracy:Int) {}
}

class AerialOverlay(context:Context):View(context) {
    var objects=emptyList<Observation>()
    var aspect=1f
    private val paint=Paint(Paint.ANTI_ALIAS_FLAG)
    override fun onDraw(canvas:Canvas) {
        super.onDraw(canvas)
        val w=min(width.toFloat(),height*aspect); val h=w/aspect
        val x=(width-w)/2; val y=(height-h)/2
        objects.forEach { o ->
            paint.color=when(o.label) { "드론"->Color.RED; "드론 의심"->0xFFFFAA40.toInt(); "새","새 의심"->Color.GREEN; else->Color.LTGRAY }
            paint.alpha=if(o.observed)255 else 110
            paint.style=Paint.Style.STROKE; paint.strokeWidth=2*resources.displayMetrics.density
            val left=x+o.rect.left*w; val top=y+o.rect.top*h
            canvas.drawRect(left,top,x+o.rect.right*w,y+o.rect.bottom*h,paint)
            paint.style=Paint.Style.FILL; paint.textSize=12*resources.displayMetrics.scaledDensity
            val text=(if(o.id>0)"#${o.id} " else "")+o.label+if(o.observed)" ${(o.score*100).roundToInt()}" else ""
            canvas.drawText(text,left.coerceAtLeast(0f),max(y+paint.textSize,top-4),paint)
        }
    }
}

class AerialAlarm(private val context:Context):java.io.Closeable {
    private val handler=Handler(Looper.getMainLooper())
    private val tone=ToneGenerator(AudioManager.STREAM_NOTIFICATION,70)
    private val vibrator=context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    var enabled=true
    private var level=0
    private var last=0L
    private var scheduled=false
    private var playingUntil=0L
    private val repeatPattern=Runnable { pulse() }
    fun observe(value:Int) {
        val promoted=value>level
        level=value; last=SystemClock.elapsedRealtime()
        if(!enabled || value==0)return
        if(!scheduled) { scheduled=true; pulse() }
        else if(promoted) {
            handler.removeCallbacks(repeatPattern)
            handler.postDelayed(repeatPattern,max(0,playingUntil-last))
        }
    }
    private fun pulse() {
        if(!enabled || level==0 || SystemClock.elapsedRealtime()-last>1000) { scheduled=false; return }
        val count=level
        val sound=when(count){1->ToneGenerator.TONE_PROP_BEEP;2->ToneGenerator.TONE_PROP_ACK;else->ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD}
        playingUntil=SystemClock.elapsedRealtime()+count*260L
        // Complete the observed event's short pattern even if the next AI frame is empty.
        repeat(count) { i -> handler.postDelayed({ if(enabled)tone.startTone(sound,130) },i*260L) }
        if(count==3)vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0,120,100,120,100,180),-1))
        handler.postDelayed(repeatPattern,when(count){1->5000L;2->3000L;else->2000L})
    }
    fun stop() { handler.removeCallbacksAndMessages(null); tone.stopTone(); vibrator.cancel(); scheduled=false; level=0 }
    override fun close() { stop(); tone.release() }
}
