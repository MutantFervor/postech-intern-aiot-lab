package com.example.activity_sensor_testing;

import static com.example.activity_sensor_testing.FeatureExtractor.extractFeatures;

import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.collection.CircularArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

// Start -> Stop -> 종료 후 결과 출력
public class TestActivity extends AppCompatActivity implements SensorEventListener {
    // 기존 변수
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGravity, mGyroscope;
    private TextView textViewTestResults;
    private boolean isTestActive = false;
    private long startTimestamp = -1, stopTimestamp = -1;
    private static final String TAG = "TestActivity";
    private Handler delayHandler = new Handler();
    private Runnable startSensorRunnable;
    private static final int WINDOW_SIZE = 60_000; // temp setting
    private static final int BUFFER_SIZE = 150;

    // CircularArray (raw data 용)
    private final CircularArray<SensorData> accelerometerBuffer = new CircularArray<>(WINDOW_SIZE);
    private final CircularArray<SensorData> gravityBuffer = new CircularArray<>(WINDOW_SIZE);
    private final CircularArray<SensorData> gyroscopeBuffer = new CircularArray<>(WINDOW_SIZE);

    // Triple-buffer (처리 용), 진행 방향: buf2 -> buf1
    private final Deque<CombinedSensorData> buf1 = new LinkedList<>();// processing
    private final Deque<CombinedSensorData> buf2 = new LinkedList<>(); // processing

    // 그 이외
    private DataProcessor dataProcessor = new DataProcessor();
    private final Handler handler = new Handler();
    private final Handler syncHandler = new Handler(); // Handler 선언
    private Handler sensorProcessingHandler; // 선언 추가
    private static final double WINDOW_DURATION = 1.0; // 1.0초
    private static final double OVERLAP_RATIO = 0.8;  // 80%
    private static final int SAMPLING_RATE_HZ = 100;  // 100Hz 샘플링 속도
    private static final int BUF2_SIZE = (int) (WINDOW_DURATION * (1 - OVERLAP_RATIO) * SAMPLING_RATE_HZ); // 30개 (buf2가 요구하는 데이터 크기 계산)
    private static final int SYNC_INTERVAL = (int) ((double) BUF2_SIZE / SAMPLING_RATE_HZ * 1000) / 2;  // SYNC_INTERVAL 계산 (밀리초 단위)
    private Server server;

    // 필드 변수 추가
    private String lastStableActivity = "unknown"; // 이전 안정 상태
    private long lastStableTime = 0; // 마지막 안정된 상태의 시간
    private int stableCount = 0; // 안정 상태 횟수 카운트
    // private static final long TRANSITION_FILTER_MS = SYNC_INTERVAL; // 전환 필터링 시간 (2초)


    // (1) onCreate
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        Button startButton = findViewById(R.id.startTestButton);
        Button stopButton = findViewById(R.id.stopTestButton);
        Button backToMainButton = findViewById(R.id.backToMainButton);
        textViewTestResults = findViewById(R.id.textViewTestResults);

        startButton.setOnClickListener(v -> startTest());
        stopButton.setOnClickListener(v -> stopTest());
        backToMainButton.setOnClickListener(v -> {
            Intent intent = new Intent(TestActivity.this, MainActivity.class);
            startActivity(intent);
            finish();
        });

        // 비동기 처리를 위한 HandlerThread 생성
        HandlerThread handlerThread = new HandlerThread("SensorProcessingThread");
        handlerThread.start();
        sensorProcessingHandler = new Handler(handlerThread.getLooper());

        server = new Server("http://192.168.0.236:5000/"); // Flask 서버 주소 입력
    }

    // (2). startTest 버튼 눌렀을 때 처리
    public void startTest() {
        if (!isTestActive) {
            int samplingRate = 10000;
            int delayMillis = 500;

            clearBuffers(); // 기존 buf 데이터 초기화
            runOnUiThread(() -> textViewTestResults.setText("Sensor will start collecting data..."));
            isTestActive = true;

            // 새로운 HandlerThread 생성 및 시작
            HandlerThread handlerThread = new HandlerThread("SensorProcessingThread");
            handlerThread.start();
            sensorProcessingHandler = new Handler(handlerThread.getLooper());

            delayHandler.postDelayed(() -> {
                // 센서 리스너 등록
                mSensorManager.registerListener(this, mAccelerometer, samplingRate);
                mSensorManager.registerListener(this, mGravity, samplingRate);
                mSensorManager.registerListener(this, mGyroscope, samplingRate);

                // 1초에 한번씩 동기화 작업 실행
                scheduleSyncTask();
            }, delayMillis); // 2초 지연
        }
    }

    // (2) - 1. 주기적으로 데이터 동기화 작업 실행
    private void scheduleSyncTask() {
        syncHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (isTestActive) {
                    timestampSync(10_000); // 동기화 작업 수행
                    syncHandler.postDelayed(this, SYNC_INTERVAL); // 다음 실행 예약
                }
            }
       }, SYNC_INTERVAL);
    }

    // (2) - 2. timestamp에서의 sync 진행
    public void timestampSync(int interval) {
        try {
            // 1. Circular Buffers 데이터를 읽고 병합
            Map<String, List<String[]>> inputData = new HashMap<>();
            inputData.put("linear", popBufferData(accelerometerBuffer));
            inputData.put("gyro", popBufferData(gyroscopeBuffer));
            inputData.put("gravity", popBufferData(gravityBuffer));

            // 2. 데이터 병합 및 동기화
            List<String[]> resultData = dataProcessor.mergeAndSyncFiles(inputData, interval);

            // 3. 결과 출력 (log 확인 용)
            for (String[] row : resultData) {
                // SYNC_INTERVAL을 1초 이하로 잡을 경우 0.01초 정도의 empty가 생기지만 어쩔 수 없음
                Log.d(TAG, "Synced data: " + Arrays.toString(row));
            }

            // timestamp sync한 Data를 buf로 보냄
            processIncomingData(resultData);
        } catch (Exception e) {
            Log.e(TAG, "Error during timestamp synchronization: " + e.getMessage());
        }
    }

    // (2) - 2 - 1. Circular Buffer 데이터를 pop하여 List<String[]>로 변환
    public List<String[]> popBufferData(CircularArray<SensorData> buffer) {
        List<String[]> data = new ArrayList<>();
        synchronized (buffer) {
            for (int i = 0; i < buffer.size(); i++) {
                SensorData sensorData = buffer.get(i);
                data.add(new String[]{
                        "", // activity_class (빈 값으로 유지, 테스트 중이므로 사용 X)
                        String.valueOf(sensorData.timestamp), String.valueOf(sensorData.x),
                        String.valueOf(sensorData.y), String.valueOf(sensorData.z)
                });
            }
            buffer.clear(); // 동기화 내에서 버퍼 초기화
        }
        return data;
    }

    private long lastProcessedTimestamp = -1; // 마지막 처리된 timestamp

    // (2) - 2 - 2. processIncomingData 수정
    public void processIncomingData(List<String[]> data) throws IOException {
        synchronized (buf1) {
            for (String[] row : data) {
                CombinedSensorData temp = CombinedSensorData.fromRawData(row);

                // 중복 방지: 마지막 처리된 timestamp보다 큰 경우에만 추가
                if (temp.timestamp > lastProcessedTimestamp) {
                    buf1.addLast(temp);
                    lastProcessedTimestamp = temp.timestamp; // 업데이트
                }
            }
            processSlidingWindow();
        }
    }

    // 슬라이딩 윈도우 처리
    private void processSlidingWindow() throws IOException {
        final int WINDOW_SIZE = (int) (WINDOW_DURATION * SAMPLING_RATE_HZ); // 윈도우 크기
        final int STEP_SIZE = (int) (WINDOW_SIZE * (1 - OVERLAP_RATIO));   // 스텝 크기

        // 임시 수정
        while (buf1.size() >= WINDOW_SIZE) {
            // 현재 윈도우 데이터를 추출
            List<CombinedSensorData> windowData = new ArrayList<>();
            Iterator<CombinedSensorData> iterator = buf1.iterator();
            for (int i = 0; i < WINDOW_SIZE && iterator.hasNext(); i++) {
                windowData.add(iterator.next());
            }

            // feature 추출
            featureExtraction(windowData);

            // STEP_SIZE만큼 데이터를 삭제 (오버랩 제외)
            for (int i = 0; i < STEP_SIZE && !buf1.isEmpty(); i++) {
                buf1.pollFirst();
            }
        }
    }

    // 피처 추출 함수
    private static final int WINDOW_SIZE_ = 5; // Majority Voting에 사용할 최근 예측의 개수
    // private static final long MIN_INTERVAL_MS = 1000; // 최소 전송 간격 (1초)
    private static final int THRESHOLD_COUNT = 5; // 행동 안정화를 위한 최소 연속 예측 횟수
    private final Queue<String> recentPredictions = new LinkedList<>();

    public void featureExtraction(List<CombinedSensorData> windowData) {
        // Extract raw features from windowData
        double[] features = extractFeatures(windowData);

        // Create instances of FeatureNormalizer and SVMPredictor
        FeatureNormalizer normalizer = new FeatureNormalizer(this);
        SVMPredictor predictor = new SVMPredictor(this, "svm_trained_model.model");

        // Normalize features
        double[] normalizedFeatures = normalizer.normalizeFeatures(features);
        if (normalizedFeatures == null) {
            System.err.println("Feature normalization failed.");
            return;
        }

        // Perform SVM prediction
        double prediction = predictor.predict(normalizedFeatures);
        int pred_int = (int) prediction;
        String activity;

        // pred_int 값에 따라 activity 문자열 설정
        switch (pred_int) {
            case 1: activity = "walking"; break;
            case 2: activity = "running"; break;
            case 3: activity = "standing"; break;
            case 4: activity = "sitting"; break;
            case 5: activity = "upstairs"; break;
            case 6: activity = "downstairs"; break;
            default: activity = "unknown"; break;
        }

        long currentTime = System.currentTimeMillis(); // 현재 시간

        // 최근 예측 추가
        if (recentPredictions.size() == WINDOW_SIZE_) {
            recentPredictions.poll(); // 가장 오래된 예측 제거
        }
        recentPredictions.add(activity);

        // Majority Voting을 통해 안정화된 활동 결정
        String majorityActivity = getMajorityActivity();

        // (1) 안정화된 상태 업데이트
        if (!majorityActivity.equals(lastStableActivity)) {
            lastStableActivity = majorityActivity;
            lastStableTime = currentTime;
            stableCount = 1; // 안정화된 카운트 초기화
            System.out.println("Activity updated to: " + lastStableActivity);
            sendPredictionToServer(lastStableActivity);
        } else {
            // 동일 활동일 경우 안정화된 상태 유지
            stableCount++;
            if (stableCount >= THRESHOLD_COUNT) {
                lastStableTime = currentTime;
                sendPredictionToServer(lastStableActivity);
                System.out.println("Stable Activity resent: " + lastStableActivity);
                // currentTime - lastStableTime >= MIN_INTERVAL_MS
            }
        }
    }

    private String getMajorityActivity() {
        Map<String, Integer> activityCounts = new HashMap<>();
        for (String activity : recentPredictions) {
            activityCounts.put(activity, activityCounts.getOrDefault(activity, 0) + 1);
        }
        return activityCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new IllegalStateException("No predictions available"))
                .getKey();
    }

    private void sendPredictionToServer(String activity) {
        // sensorProcessingHandler.post(() -> {
        server.sendActivity(activity, new Server.ServerCallback() {
            @Override
            public void onSuccess(String message) {
                Log.d(TAG, "Server Response: " + message);
            }

            @Override
            public void onFailure(String error) {
                Log.e(TAG, "Server Error: " + error);
            }
        });
        // });
    }

    // (2) - 7. buf 내용 제거 함수
    public void clearBuffers() {
        accelerometerBuffer.clear();
        gravityBuffer.clear();
        gyroscopeBuffer.clear();
        synchronized (buf2) {
            buf2.clear();
        }
        synchronized (buf1) {
            buf1.clear();
        }
    }

    // (3) sensor 작동 시 처리 - 각 sensor의 buf에 정보 저장
    @Override
    public void onSensorChanged(SensorEvent event) {
        long timestamp = event.timestamp / 1000L; // 나노초 → 마이크로초 변환
        float[] values = event.values.clone();

        if (startTimestamp == -1) {
            startTimestamp = timestamp;
            runOnUiThread(() -> textViewTestResults.setText("Sensor is running..."));
        }
        stopTimestamp = timestamp;
        SensorData sensorData = new SensorData(timestamp, values[0], values[1], values[2]);

        // 버퍼에 동기화하여 추가
        sensorProcessingHandler.post(() -> {
            synchronized (getBufferForSensor(event.sensor.getType())) {
                getBufferForSensor(event.sensor.getType()).addLast(sensorData);
            }
        });
    }

    // CircularArray 동기화 문제 해결: 센서별 버퍼 반환
    public CircularArray<SensorData> getBufferForSensor(int sensorType) {
        switch (sensorType) {
            case Sensor.TYPE_LINEAR_ACCELERATION:
                return accelerometerBuffer;
            case Sensor.TYPE_GRAVITY:
                return gravityBuffer;
            case Sensor.TYPE_GYROSCOPE:
                return gyroscopeBuffer;
            default:
                throw new IllegalArgumentException("Unknown sensor type: " + sensorType);
        }
    }

    // (4) stop 버튼 눌렀을 때의 처리
    public void stopTest() {
        if (isTestActive) {
            mSensorManager.unregisterListener(this, mAccelerometer);
            mSensorManager.unregisterListener(this, mGravity);
            mSensorManager.unregisterListener(this, mGyroscope);
            isTestActive = false;
            startTimestamp = -1;

            if (startSensorRunnable != null) {
                delayHandler.removeCallbacks(startSensorRunnable);
            }

            // 동기화 작업 중단
            syncHandler.removeCallbacksAndMessages(null);
            sensorProcessingHandler.getLooper().quitSafely();
            runOnUiThread(() -> textViewTestResults.setText("Test stopped."));
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        server.startStreaming(textViewTestResults);
    }

    @Override
    protected void onStop() { super.onStop(); }
    @Override
    protected void onDestroy() { super.onDestroy(); }
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }
}