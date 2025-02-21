package com.example.activity_sensor_testing;
import androidx.annotation.NonNull;

public class SensorData {
    public long timestamp;
    public double x, y, z;

    public SensorData(long timestamp, double x, double y, double z) {
        this.timestamp = timestamp;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @NonNull
    @Override
    public String toString() {
        return "Timestamp=" + timestamp + ", X=" + x + ", Y=" + y + ", Z=" + z;
    }
}