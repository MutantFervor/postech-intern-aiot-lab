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
    private TextView textViewValues;
    private boolean isSensorsActive = false;
    private int activity_class = -1;

    // temp
    private String accelerometerData, gravityData, gyroscopeData = "";
    private String TAG;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Button setting
        Button StartStopButton = findViewById(R.id.StartStopButton);
        // Button PauseResumeButton = findViewById(R.id.PauseResumeButton);
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
                textViewValues.setText("Please select an activity.");
            }
        });

        // 2. DiscardButton
        DiscardButton.setOnClickListener(v -> {
            textViewValues.setText("Discarded.");
            // 중단 및 삭제. 한 Start -> Stop 내의 모든 기록을 삭제한다.
            // (1) Start에서의 TimeStamp를 기록

            // (2) Stop을 눌렀을 당시의 TimeStamp를 기록
            // (3) 미리 생성되어 있던 activity의 파일에 접근
            // (4) Start to Stop 사이 TimeStamp를 모두 삭제
        });


        // Sensor setting
        mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        // TAG
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

    // 임시 실험을 위한 코드로 작성
    @Override
    public void onSensorChanged(SensorEvent event) {
        long timestamp = event.timestamp / 1_000L; // conversion to microsec (확인 필요)
        float[] value = event.values.clone();

        // 센서 타입별 데이터 포맷팅 및 할당
        String formattedData = String.format(
                "%d, %d, %.9e, %.9e, %.9e\n", activity_class, timestamp, value[0], value[1], value[2]
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

    public void findSelectedRadioButton(String radioButtonName) {
        String[] activities = {"Other", "Walking", "Running", "Standing", "Sitting", "Upstairs", "Downstairs"};
        activity_class = java.util.Arrays.asList(activities).indexOf(radioButtonName);
    }

    public void startSensor() {
        int samplingRate = 10000;
        int delayMillis = 5000;  // 5초 딜레이 (밀리초)

        // Delayed Start 구현
        textViewValues.setVisibility(TextView.VISIBLE); // 센서 데이터 표시
        textViewValues.setText("Sensor will start in 5 seconds...");
        isSensorsActive = true;
        new android.os.Handler().postDelayed(() -> {
            // 센서 리스너 등록 (5초 후 실행)
            mSensorManager.registerListener(this, mAccelerometer, samplingRate);
            mSensorManager.registerListener(this, mGravity, samplingRate);
            mSensorManager.registerListener(this, mGyroscope, samplingRate);
        }, delayMillis); // 5초 딜레이
    }

    public void stopSensor() {
        // 센서 실행 종료
        mSensorManager.unregisterListener(this, mAccelerometer);
        mSensorManager.unregisterListener(this, mGravity);
        mSensorManager.unregisterListener(this, mGyroscope);

        isSensorsActive = false;
        textViewValues.setVisibility(TextView.GONE); // 숨김 처리

        // temp
        clearSensorData();

        // 앞에서의 5초간의 데이터를 삭제. (1초에 100개가 저장되므로 500개의 data를 삭제하는 형식)
        earlyStop();
    }

    // 앞에서의 5초간의 데이터를 삭제하는 함수
    public void earlyStop() {
        // temp empty
    }

    // temp
    public void clearSensorData() {
        accelerometerData = "";
        gravityData = "";
        gyroscopeData = "";
    }
}