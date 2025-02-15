package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import android.graphics.Color


class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Python の初期化
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.apply {
            isGpu.setOnCheckedChangeListener { buttonView, isChecked ->
                cameraExecutor.submit {
                    detector?.restart(isGpu = isChecked)
                }
                if (isChecked) {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.orange))
                } else {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector?.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
        }
    }

    private fun saveBitmapToFile(bitmap: Bitmap?): String {
        if (bitmap == null) {
            Log.e(TAG, "Bitmap is null in saveBitmapToFile")
            throw IllegalArgumentException("Bitmap が null です")
        }
        val cacheDir = applicationContext.cacheDir
        val file = File(cacheDir, "captured_image.jpg")
        FileOutputStream(file).use { outStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outStream)
        }
        Log.d(TAG, "Image saved at: ${file.absolutePath}")
        return file.absolutePath
    }

    /**
     * Python の process_detections 関数を呼び出し、結果をコールバックで返す
     *
     * @param jsonData 検出結果の JSON 文字列
     * @param imagePath 画像ファイルのパス
     * @param threshold 二値化の閾値（例: 128）
     * @param callback 判定結果（List<Int>）を受け取るコールバック
     */
    private fun runTargetJudgement(jsonData: String, imagePath: String, threshold: Int, callback: (List<Int>?) -> Unit) {
        Thread {
            try {
                val py = Python.getInstance()
                val module: PyObject = py.getModule("target_judgement")
                val resultPy: PyObject = module.callAttr("process_detections", jsonData, imagePath, threshold)
                val resultMap = resultPy.asMap() as Map<String, PyObject>
                val resultObj = resultMap["results"]
                if (resultObj == null) {
                    Log.e("Debug", "[PYTHON LOG][Kotlin] 'results' key is null in resultMap")
                    runOnUiThread { callback(null) }
                    return@Thread
                }
                // Python側が返すリストをそのままJavaのListに変換する
                val pyList: List<PyObject> = resultObj.asList()
                val resultList: List<Int> = pyList.map { it.toInt() }
                Log.d("Debug", "[PYTHON LOG][Kotlin] Received judgement result: $resultList")
                runOnUiThread { callback(resultList) }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread { callback(null) }
            }
        }.start()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        Log.d("Debug", "[PYTHON LOG][Kotlin] boundingBoxes: $boundingBoxes")
        runOnUiThread {
            // 初回描画：各ボックスは生成時のデフォルト色（arrowはデフォルトで赤色など）になっている
            binding.overlay.setResults(boundingBoxes)
            binding.overlay.invalidate()

            // 推論時間を表示
            binding.inferenceTime.text = "Inference: ${inferenceTime}ms"

            // JSON生成：すべての検出結果を含む
            val predictionsArray = JSONArray()
            for (box in boundingBoxes) {
                val obj = JSONObject().apply {
                    put("x", box.cx)
                    put("y", box.cy)
                    put("width", box.w)
                    put("height", box.h)
                    put("class", box.clsName) // 例："arrow", "targetw", "targetb"
                }
                predictionsArray.put(obj)
            }
            val jsonObj = JSONObject().apply { put("predictions", predictionsArray) }
            val jsonData = jsonObj.toString()
            Log.d("Debug", "[PYTHON LOG][Kotlin] Generated JSON: $jsonData")

            // 画像を保存
            val imagePath = saveBitmapToFile(binding.viewFinder.bitmap)
            Log.d("Debug", "[PYTHON LOG][Kotlin] Saved image file path: $imagePath")

            // Python 側の判定処理を呼び出す
            runTargetJudgement(jsonData, imagePath, 128) { result ->
                if (result != null && result.isNotEmpty()) {
                    // 結果リストに -1 が含まれている場合は、正しい判定が行われなかったとみなし、「なし」と表示
                    if (result.any { it == -1 }) {
                        binding.judgementResultText.text = "判定結果: なし"
                    } else {
                        // arrow の判定結果は二値リスト（的中なら1、外れなら0）のはず
                        val hitCount = result.sum()
                        binding.judgementResultText.text = "的中本数: $hitCount 本"

                        // boundingBoxes のうち、arrow のみを抽出して、判定結果に合わせた色を設定する
                        val arrowBoxes = boundingBoxes.filter { it.clsName == "arrow" }
                        if (arrowBoxes.size == result.size) {
                            for (i in arrowBoxes.indices) {
                                if (result[i] == 1) {
                                    // 的中：黄緑色
                                    arrowBoxes[i].color = Color.parseColor("#ADFF2F")
                                } else {
                                    // 外れ：赤色
                                    arrowBoxes[i].color = Color.RED
                                }
                            }
                        } else {
                            Log.e("Debug", "arrowの件数が一致しません。arrowBoxes.size=${arrowBoxes.size}, result.size=${result.size}")
                        }
                    }
                    // targetw と targetb のバウンディングボックスは常に紺色にする
                    boundingBoxes.forEach { box ->
                        if (box.clsName == "targetw" || box.clsName == "targetb") {
                            box.color = Color.parseColor("#000080")  // 紺色
                        }
                    }
                    // 更新後、オーバーレイを再描画
                    binding.overlay.setResults(boundingBoxes)
                    binding.overlay.invalidate()
                    Log.d("Debug", "[PYTHON LOG][Kotlin] Received judgement result: $result")
                } else {
                    Log.e("Debug", "判定結果の取得に失敗")
                    binding.judgementResultText.text = "判定結果: なし"
                }
            }
        }
    }
}
