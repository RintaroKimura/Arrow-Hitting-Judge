buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        // Android Gradle Plugin の依存関係
        classpath("com.android.tools.build:gradle:8.8.0")
        // Chaquopy の依存関係を追加
        classpath("com.chaquo.python:gradle:16.0.0")
    }
}

plugins {
    id("com.android.application") version "8.8.0" apply false
    id("org.jetbrains.kotlin.android") version "2.0.20" apply false
}