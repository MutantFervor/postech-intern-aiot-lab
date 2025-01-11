package com.example.activity_sensor;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGravity, mGyroscope;
    private TextView textViewValues;
    private int selectedRadioButtonId = -1;
    private boolean isSensorsActive = false;
    private int activity_class = -1;
    private File activityFolder;
    private File linearFile, gravityFile, gyroscopeFile;
    private FileWriter linearWriter, gravityWriter, gyroscopeWriter;
    private long startTimestamp = -1, stopTimestamp = -1;
    private android.os.Handler delayHandler = new android.os.Handler();
    private Runnable startSensorRunnable;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Button setting
        Button StartStopButton = findViewById(R.id.StartStopButton);
        Button DiscardButton = findViewById(R.id.DiscardButton);
        RadioGroup activityRadioGroup = findViewById(R.id.activityRadioGroup);
        textViewValues = findViewById(R.id.textViewValues);

        // RadioGroup 상태 변경 리스너
        activityRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            int lastSelectedRadioButton = selectedRadioButtonId;
            if (isSensorsActive) {
                showStopFirstPopup();
                activityRadioGroup.check(lastSelectedRadioButton);
            } else {
                RadioButton selectedRadioButton = findViewById(checkedId);
                selectedRadioButtonId = checkedId;
                if (selectedRadioButton != null) {
                    findSelectedRadioButton(selectedRadioButton.getText().toString()); // 선택된 라디오 버튼의 텍스트 저장
                }
            }
        });

        // Button setOnclickListener
        StartStopButton.setOnClickListener(v -> {
            if (isSensorsActive) {
                stopSensor();
            } else if (activity_class != -1) { // 라디오 버튼 선택 확인
                startSensor();
            } else {
                textViewValues.setText("Please select an activity.");
            }
        });

        // ****
        DiscardButton.setOnClickListener(v -> {
            if (isSensorsActive) {
                // 현재 Discard 시점의 timestamp 기록
                stopTimestamp = System.currentTimeMillis() * 1000L; // 현재 시간을 마이크로초로 변환

                // 각 파일에 대해 startTimestamp ~ stopTimestamp 데이터를 삭제
                discardCycleData(linearFile);
                discardCycleData(gravityFile);
                discardCycleData(gyroscopeFile);

                textViewValues.setText("Data discarded for the current cycle.");

                // Cycle 초기화 (선택 사항)
                startTimestamp = -1;
                stopTimestamp = -1;
            } else {
                textViewValues.setText("No active sensor data to discard.");
            }
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

    // 선택 된 RadioButton을 찾는 함수
    public void findSelectedRadioButton(String radioButtonName) {
        String[] activities = {"Other", "Walking", "Running", "Standing", "Sitting", "Upstairs", "Downstairs"};
        activity_class = java.util.Arrays.asList(activities).indexOf(radioButtonName);
    }

    // 센서를 시작하는 함수
    public void startSensor() {
        int samplingRate = 10000;
        int delayMillis = 5000;  // 5초 딜레이 (밀리초)

        textViewValues.setVisibility(TextView.VISIBLE); // 센서 데이터 표시
        textViewValues.setText("Sensor will start in 5 seconds...");
        isSensorsActive = true;

        if (activity_class != -1) {
            activityFolder = new File(getExternalFilesDir(null), String.valueOf(activity_class));
            if (!activityFolder.exists()) {
                activityFolder.mkdirs();
            }

            try {
                linearFile = new File(activityFolder, "linear.csv");
                gravityFile = new File(activityFolder, "gravity.csv");
                gyroscopeFile = new File(activityFolder, "gyro.csv");

                linearWriter = new FileWriter(linearFile, true); // true: append mode
                gravityWriter = new FileWriter(gravityFile, true);
                gyroscopeWriter = new FileWriter(gyroscopeFile, true);

            } catch (IOException e) {
                e.printStackTrace();
                textViewValues.setText("Error creating files.");
            }
        }

        // Runnable 정의
        startSensorRunnable = () -> {
            mSensorManager.registerListener(this, mAccelerometer, samplingRate);
            mSensorManager.registerListener(this, mGravity, samplingRate);
            mSensorManager.registerListener(this, mGyroscope, samplingRate);
        };

        // Handler로 지연 작업 등록
        delayHandler.postDelayed(startSensorRunnable, delayMillis);
    }

    // event 감지 시에 센서의 행동
    @Override
    public void onSensorChanged(SensorEvent event) {
        long timestamp = event.timestamp / 1000L; // conversion to microsec (확인 필요)
        float[] value = event.values.clone();

        // timestamp 설정
        if (startTimestamp == -1) {
            startTimestamp = timestamp; // 첫 이벤트의 timestamp 저장
            textViewValues.setText("Start timestamp recorded: " + startTimestamp);
        }
        stopTimestamp = timestamp;

        // 센서 타입별 데이터 포맷팅 및 할당
        String formattedData = String.format(
                "%d, %d, %.9e, %.9e, %.9e\n", activity_class, timestamp, value[0], value[1], value[2]
        );

        try {
            if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION && linearWriter != null) {
                linearWriter.write(formattedData);
            } else if (event.sensor.getType() == Sensor.TYPE_GRAVITY && gravityWriter != null) {
                gravityWriter.write(formattedData);
            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE && gyroscopeWriter != null) {
                gyroscopeWriter.write(formattedData);
            }
            textViewValues.setText("Data is Writing to csv file...");
        } catch (IOException e) {
            e.printStackTrace();
            textViewValues.setText("Error writing sensor data.");
        }
    }

    // 센서를 종료하는 함수
    public void stopSensor() {
        // Delay Start 타이머 작업 취소
        if (startSensorRunnable != null) {
            delayHandler.removeCallbacks(startSensorRunnable);
        }

        mSensorManager.unregisterListener(this, mAccelerometer);
        mSensorManager.unregisterListener(this, mGravity);
        mSensorManager.unregisterListener(this, mGyroscope);
        isSensorsActive = false;

        // Close file writers
        try {
            if (linearWriter != null) {
                linearWriter.close();
                linearWriter = null; // 닫은 후 null로 설정
            }
            if (gravityWriter != null) {
                gravityWriter.close();
                gravityWriter = null; // 닫은 후 null로 설정
            }
            if (gyroscopeWriter != null) {
                gyroscopeWriter.close();
                gyroscopeWriter = null; // 닫은 후 null로 설정
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // earlyStop 과정 실행 (delay Start 중 실행되면 earlyStop 실행 X)
        if (startTimestamp != -1) {
            earlyStop();
        }

        // cycle 종료 후 변수 초기화
        startTimestamp = -1;
        stopTimestamp = -1;

        textViewValues.setText("Stopped.");
    }

    // (종료 시) 뒤에서부터 500개의 데이터를 삭제하는 함수 ****
    public void earlyStop() {
        // 각 센서 파일에 대해 최근 500개의 데이터를 삭제
        if (activityFolder == null || !activityFolder.exists()) {
            textViewValues.setText("No activity folder exists.");
            return;
        }

        // 파일에서 현재 cycle의 데이터만 삭제
        deleteCycleData(linearFile, 500);
        deleteCycleData(gravityFile, 500);
        deleteCycleData(gyroscopeFile, 500);

        textViewValues.setText("Early stop completed. Last 500 entries from current cycle deleted.");
    }

    // ****
    private void deleteCycleData(File file, int n) {
        if (file == null || !file.exists()) return;

        List<String> allLines = new ArrayList<>();
        List<String> filteredLines = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;

            while ((line = reader.readLine()) != null) {
                allLines.add(line);

                // 데이터의 타임스탬프 추출
                String[] tokens = line.split(",");
                if (tokens.length < 2) continue;

                long timestamp = Long.parseLong(tokens[1].trim());

                // 5초 이내: startTimestamp와 stopTimestamp 사이의 데이터는 제외
                if (stopTimestamp - startTimestamp <= 5_000_000L) {
                    if (timestamp < startTimestamp || timestamp > stopTimestamp) {
                        filteredLines.add(line);
                    }
                } else { // 5초 초과: 뒤에서 n개의 데이터만 삭제
                    filteredLines.add(line); // 모든 데이터를 일단 추가
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 5초 초과일 경우 뒤에서 n개의 데이터만 삭제
        if (stopTimestamp - startTimestamp > 5_000_000L) {
            if (filteredLines.size() > n) {
                filteredLines = filteredLines.subList(0, filteredLines.size() - n);
            } else {
                filteredLines.clear();
            }
        }

        // 파일에 덮어쓰기
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            for (String line : filteredLines) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // discardCycle data
    private void discardCycleData(File file) {
        // Delay Start 타이머 작업 취소
        if (startSensorRunnable != null) {
            delayHandler.removeCallbacks(startSensorRunnable);
        }

        mSensorManager.unregisterListener(this, mAccelerometer);
        mSensorManager.unregisterListener(this, mGravity);
        mSensorManager.unregisterListener(this, mGyroscope);
        isSensorsActive = false;

        // Close file writers
        try {
            if (linearWriter != null) {
                linearWriter.close();
                linearWriter = null; // 닫은 후 null로 설정
            }
            if (gravityWriter != null) {
                gravityWriter.close();
                gravityWriter = null; // 닫은 후 null로 설정
            }
            if (gyroscopeWriter != null) {
                gyroscopeWriter.close();
                gyroscopeWriter = null; // 닫은 후 null로 설정
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (file == null || !file.exists()) return;

        List<String> filteredLines = new ArrayList<>();
        boolean inCurrentCycle = false;

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;

            while ((line = reader.readLine()) != null) {
                // 데이터의 타임스탬프 추출
                String[] tokens = line.split(",");
                if (tokens.length < 2) continue;

                long timestamp = Long.parseLong(tokens[1].trim());

                if (startTimestamp == -1 || stopTimestamp == -1) return;
                if (timestamp < startTimestamp || timestamp > stopTimestamp) {
                    filteredLines.add(line);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 파일에 덮어쓰기
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            for (String line : filteredLines) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 실행 도중 RadioButton 변경 시도 시
    private void showStopFirstPopup() {
        Toast.makeText(this, "Please stop the current activity before changing.", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        // empty
    }
}
