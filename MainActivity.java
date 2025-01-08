package com.example.activity_sensor;

import androidx.appcompat.app.AppCompatActivity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGravity, mGyroscope;
    private TextView textViewValues;
    private boolean isSensorsActive = false;
    private int activity_class = -1;
    private FirebaseStorage firebaseStorage;
    private StorageReference storageReference;
    private File accelerometerFile, gravityFile, gyroscopeFile;

    // temp
    private String accelerometerData, gravityData, gyroscopeData = "";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Database setting
        // Firebase Storage 초기화
        firebaseStorage = FirebaseStorage.getInstance();
        storageReference = firebaseStorage.getReference();

        // Button setting
        Button StartStopButton = findViewById(R.id.StartStopButton);
        Button DiscardButton = findViewById(R.id.DiscardButton);
        RadioGroup activityRadioGroup = findViewById(R.id.activityRadioGroup);
        textViewValues = findViewById(R.id.textViewValues);

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
                setupActivityFiles();
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
    }

    @Override
    public void onStart() {
        super.onStart();
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    // 임시 실험을 위한 코드로 작성
    @Override
    public void onSensorChanged(SensorEvent event) {
        long timestamp = event.timestamp / 1_000L; // conversion to microsecond
        float[] value = event.values.clone();

        // 센서 타입별 데이터 포맷팅 및 할당
        String formattedData = String.format(
                "%d, %d, %.9e, %.9e, %.9e\n", activity_class, timestamp, value[0], value[1], value[2]
        );

        try {
            if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
                appendDataToFile(accelerometerFile, formattedData);
                accelerometerData = formattedData;
            } else if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
                appendDataToFile(gravityFile, formattedData);
                gravityData = formattedData;
            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                appendDataToFile(gyroscopeFile, formattedData);
                gyroscopeData = formattedData;
            }
        } catch (IOException e) {
            Log.e("SensorData", "Failed to write data: " + e.getMessage());
        }

        // (temp) Combine all sensor data and display
        String combinedData = accelerometerData + gravityData + gyroscopeData;
        textViewValues.setText(combinedData);
    }

    public void findSelectedRadioButton(String radioButtonName) {
        String[] activities = {"Other", "Walking", "Running", "Standing", "Sitting", "Upstairs", "Downstairs"};
        activity_class = java.util.Arrays.asList(activities).indexOf(radioButtonName);
    }

    public void setupActivityFiles() {
        File activityFolder = new File(getFilesDir(), String.valueOf(activity_class));
        if (!activityFolder.exists()) {
            activityFolder.mkdirs();
        }

        accelerometerFile = createOrOpenFile(activityFolder, "linear.csv");
        gravityFile = createOrOpenFile(activityFolder, "gravity.csv");
        gyroscopeFile = createOrOpenFile(activityFolder, "gyro.csv");
    }

    public File createOrOpenFile(File folder, String fileName) {
        File file = new File(folder, fileName);
        if (!file.exists()) {
            try (FileWriter writer = new FileWriter(file)) {
                writer.append("activity_class, timestamp, value_x, value_y, value_z\n");
            } catch (IOException e) {
                Log.e("FileSetup", "Failed to create file: " + e.getMessage());
            }
        }
        return file;
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
        textViewValues.setText("Sensor will start in 5 seconds...");
        // textViewValues.setVisibility(TextView.GONE); // 숨김 처리

        // temp
        clearSensorData();

        // 앞에서의 5초간의 데이터를 삭제. (1초에 100개가 저장되므로 500개의 data를 삭제하는 형식)
        earlyStop();
    }

    private void appendDataToFile(File file, String data) throws IOException {
        try (FileWriter writer = new FileWriter(file, true)) {
            writer.append(data);
        }
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

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        // Not used
    }
}