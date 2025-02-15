package com.surendramaran.yolov8tflite
import android.graphics.Color

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String,
    var color: Int = Color.TRANSPARENT
)