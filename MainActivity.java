package com.example.activity_sensor;

import androidx.appcompat.app.AppCompatActivity;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGravity, mGyroscope;

    // temp
    private TextView textViewValues; // temp
    private String accelerometerData = "";
    private String gravityData = "";
    private String gyroscopeData = "";

    private boolean isSensorsActive = false;
    private int activity_class = -1;

    private String TAG;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Button setting
        Button StartStopButton = findViewById(R.id.StartStopButton);
        Button PauseResumeButton = findViewById(R.id.PauseResumeButton);
        Button DiscardButton = findViewById(R.id.DiscardButton);
        textViewValues = findViewById(R.id.textViewValues);

        RadioGroup activityRadioGroup = findViewById(R.id.activityRadioGroup);

        // RadioGroup 상태 변경 리스너
        activityRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            RadioButton selectedRadioButton = findViewById(checkedId);
            if (selectedRadioButton != null) {
                findSelectedRadioButton(selectedRadioButton.getText().toString()); // 선택된 라디오 버튼의 텍스트 저장
            }
        });

        // Button setOnclickListener
        // 1. StartStopButton
        StartStopButton.setOnClickListener(v -> {
            if (isSensorsActive) {
                stopSensor();
            } else if (activity_class != -1) { // 라디오 버튼 선택 확인
                startSensor();
            } else {
                textViewValues.setText("Please select an activity."); // 사용자에게 알림
            }
        });

        // 2. PauseResumeButton

        // 3. DiscardButton


        // Sensor setting
        mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        TAG = MainActivity.class.getName() + " " + Thread.currentThread().getName();
    }

    @Override
    public void onStart() {
        super.onStart();
        Log.d(TAG, "onStart");
    }

    @Override
    public void onStop() {
        super.onStop();
        Log.d(TAG, "onStop");
    }

    public void findSelectedRadioButton(String radioButtonName) {
        String[] activities = {"Walking", "Running", "Standing", "Sitting", "Upstairs", "Downstairs"};
        activity_class = java.util.Arrays.asList(activities).indexOf(radioButtonName);
    }

    public void startSensor() {
        int samplingRate = 10000;

        // 센서 실행 시작
        mSensorManager.registerListener(this, mAccelerometer, samplingRate);
        mSensorManager.registerListener(this, mGravity, samplingRate);
        mSensorManager.registerListener(this, mGyroscope, samplingRate);

        isSensorsActive = true;
        textViewValues.setVisibility(TextView.VISIBLE); // temp
    }

    public void stopSensor() {
        // 센서 실행 종료
        mSensorManager.unregisterListener(this, mAccelerometer);
        mSensorManager.unregisterListener(this, mGravity);
        mSensorManager.unregisterListener(this, mGyroscope);

        isSensorsActive = false;
        textViewValues.setVisibility(TextView.GONE); // temp
        clearSensorData();
    }

    private void clearSensorData() {
        accelerometerData = "";
        gravityData = "";
        gyroscopeData = "";
        textViewValues.setText(""); // Clear displayed data
    }

    // 임시 실험을 위한 코드로 작성
    @Override
    public void onSensorChanged(SensorEvent event) {
        int timestamp = (int)(event.timestamp / 1_000_000L); // conversion to microsec (확인 필요)
        float value_x = event.values[0];
        float value_y = event.values[1];
        float value_z = event.values[2];

        // 센서 타입별 데이터 포맷팅 및 할당
        String formattedData = String.format(
                "%d, %d, X: %.9e, Y: %.9e, Z: %.9e\n", activity_class, timestamp, value_x, value_y, value_z
        );

        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            accelerometerData = formattedData;
        } else if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
            gravityData = formattedData;
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroscopeData = formattedData;
        }

        // (temp) Combine all sensor data and display
        String combinedData = accelerometerData + gravityData + gyroscopeData;
        textViewValues.setText(combinedData);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        // empty
    }
}