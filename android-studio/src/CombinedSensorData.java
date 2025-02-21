package com.example.activity_sensor_testing;
import static android.content.ContentValues.TAG;

import android.util.Log;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CombinedSensorData {
    public long timestamp;
    public double linear_x, linear_y, linear_z;
    public double gyro_x, gyro_y, gyro_z;
    public double gravity_x, gravity_y, gravity_z;

    public CombinedSensorData(long timestamp, double linear_x, double linear_y, double linear_z,
                              double gyro_x, double gyro_y, double gyro_z,
                              double gravity_x, double gravity_y, double gravity_z) {
        this.timestamp = timestamp;
        this.linear_x = linear_x;
        this.linear_y = linear_y;
        this.linear_z = linear_z;
        this.gyro_x = gyro_x;
        this.gyro_y = gyro_y;
        this.gyro_z = gyro_z;
        this.gravity_x = gravity_x;
        this.gravity_y = gravity_y;
        this.gravity_z = gravity_z;
    }

    @NonNull
    @Override
    public String toString() {
        return "Timestamp=" + timestamp +
                ", Linear[x=" + linear_x + ", y=" + linear_y + ", z=" + linear_z + "]" +
                ", Gyro[x=" + gyro_x + ", y=" + gyro_y + ", z=" + gyro_z + "]" +
                ", Gravity[x=" + gravity_x + ", y=" + gravity_y + ", z=" + gravity_z + "]";
    }

    public static CombinedSensorData fromRawData(String[] row) {
        try {
            // 타임스탬프와 센서 값 추출
            long timestamp = Long.parseLong(row[0]);
            double linear_x = Double.parseDouble(row[1]);
            double linear_y = Double.parseDouble(row[2]);
            double linear_z = Double.parseDouble(row[3]);
            double gyro_x = Double.parseDouble(row[4]);
            double gyro_y = Double.parseDouble(row[5]);
            double gyro_z = Double.parseDouble(row[6]);
            double gravity_x = Double.parseDouble(row[7]);
            double gravity_y = Double.parseDouble(row[8]);
            double gravity_z = Double.parseDouble(row[9]);

            // CombinedSensorData 생성 후 반환
            return new CombinedSensorData(
                    timestamp,
                    linear_x, linear_y, linear_z,
                    gyro_x, gyro_y, gyro_z,
                    gravity_x, gravity_y, gravity_z
            );
        } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
            Log.e(TAG, "Error converting row to CombinedSensorData: " + Arrays.toString(row), e);
            return null; // 변환 실패 시 null 반환
        }
    }
}