package com.example.activity_sensor_testing;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class DataProcessor {

    // 1. 파일의 timestamp 조정, 보간 및 병합
    public List<String[]> mergeAndSyncFiles(Map<String, List<String[]>> inputData, int interval) {
        if (interval <= 0) {
            interval = 10000;
        }

        List<String> files = Arrays.asList("linear", "gyro", "gravity");

        // 동기화 단계
        Map<String, Queue<String[]>> buffers = new HashMap<>();
        for (String file : files) {
            buffers.put(file, new LinkedList<>(inputData.get(file))); // 원본 데이터로 초기화
        }

        // 결과 버퍼
        List<String[]> syncedData = new ArrayList<>();
        String[] linear = buffers.get("linear").poll();
        String[] gyro = buffers.get("gyro").poll();
        String[] gravity = buffers.get("gravity").poll();

        // 모든 버퍼가 데이터가 있는 동안 동기화 및 병합
        while (buffers.values().stream().allMatch(queue -> !queue.isEmpty())) {
            // 유효하지 않은 데이터를 건너뛰기
            while (!isValidLine(linear) || !isValidLine(gyro) || !isValidLine(gravity)) {
                if (!isValidLine(linear)) {
                    linear = buffers.get("linear").poll();
                } else if (!isValidLine(gyro)) {
                    gyro = buffers.get("gyro").poll();
                } else {
                    gravity = buffers.get("gravity").poll();
                }
            }

            if (linear == null || gyro == null || gravity == null) {
                break;
            }

            // 타임스탬프가 동일한 경우 병합
            if (isSameTimestamp(Long.parseLong(linear[1]), Long.parseLong(gyro[1]), Long.parseLong(gravity[1]))) {
                syncedData.add(mergeRows(linear, gyro, gravity));
                linear = buffers.get("linear").poll();
                gyro = buffers.get("gyro").poll();
                gravity = buffers.get("gravity").poll();
            } else {
                // 가장 작은 타임스탬프를 가진 센서 데이터 제거
                int minIndex = findMinTimestamp(Long.parseLong(linear[1]), Long.parseLong(gyro[1]), Long.parseLong(gravity[1]));
                String minFile = files.get(minIndex);
                buffers.get(minFile).poll();
            }
        }

        // 타임스탬프 조정 및 보간
        List<String[]> finalResult = new ArrayList<>();
        for (String file : files) {
            List<String[]> adjustedData = adjustTimestamps(syncedData, interval); // 타임스탬프 조정
            List<String[]> interpolatedData = interpolation(adjustedData, Arrays.asList("x", "y", "z")); // 보간 작업
            finalResult.addAll(interpolatedData);
        }

        return finalResult;
    }


    // 데이터 유효성 검사
    public boolean isValidLine(String[] line) {
        if (line == null || line.length != 5) { // 필드 개수가 5개가 아닌 경우
            return false;
        }
        try {
            double timestamp = Double.parseDouble(line[1]);
            if (timestamp % 1 != 0 || timestamp < 100000) { // timestamp가 정수로 변환 가능하고 6자리 이상인지 확인
                return false;
            }
        } catch (NumberFormatException e) { // 변환 불가능한 경우
            return false;
        }
        return true;
    }

    // 동일 타임스탬프 확인
    public boolean isSameTimestamp(long t1, long t2, long t3) {
        String key1 = String.valueOf(t1).substring(0, String.valueOf(t1).length() - 4);
        String key2 = String.valueOf(t2).substring(0, String.valueOf(t2).length() - 4);
        String key3 = String.valueOf(t3).substring(0, String.valueOf(t3).length() - 4);
        return key1.equals(key2) && key1.equals(key3);
    }

    // 최소 값을 가진 타임스탬프 찾기
    public int findMinTimestamp(long t1, long t2, long t3) {
        long[] timestamps = {t1, t2, t3};
        int minIndex = 0;
        for (int i = 1; i < timestamps.length; i++) {
            if (timestamps[i] < timestamps[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    // 보간 함수
    public List<String[]> interpolation(List<String[]> df, List<String> columns) {
        Map<String, List<Double>> numericData = new HashMap<>();
        for (String column : columns) {
            numericData.put(column, new ArrayList<>());
            for (String[] row : df) {
                try {
                    numericData.get(column).add(Double.parseDouble(row[getColumnIndex(column)]));
                } catch (NumberFormatException e) {
                    numericData.get(column).add(Double.NaN);
                }
            }
        }

        for (String column : columns) {
            List<Double> data = numericData.get(column);
            // (1) 결측값을 골라내 NaN 변환 후 1차 보간
            interpolate(data);
            // IQR을 적용하여 이상치 제거
            double Q1 = getQuantile(data, 0.25);
            double Q3 = getQuantile(data, 0.75);
            double IQR = Q3 - Q1;

            double lowerBound = Q1 - 300.0 * IQR;
            double upperBound = Q3 + 300.0 * IQR;

            for (int i = 0; i < data.size(); i++) {
                double value = data.get(i);
                if (value < lowerBound || value > upperBound) {
                    data.set(i, Double.NaN);
                }
            }

            // 최종 선형 보간
            interpolate(data);

            // Update the original data
            for (int i = 0; i < data.size(); i++) {
                df.get(i)[getColumnIndex(column)] = Double.toString(data.get(i));
            }
        }
        return df;
    }

    // 타임스탬프 조정 함수
    public List<String[]> adjustTimestamps(List<String[]> dataFrame, int interval) {
        List<String[]> df = new ArrayList<>(dataFrame);
        double adjustment = 0;
        for (int i = 1; i < df.size(); i++) {
            double expected = Double.parseDouble(df.get(i - 1)[1]) + interval;
            double current = Double.parseDouble(df.get(i)[1]) - adjustment;
            if (current > expected) { // 간극 조정을 위해
                adjustment += current - expected;
                df.get(i)[1] = Double.toString(expected);
            } else {
                df.get(i)[1] = Double.toString(current);
            }
        }
        return df;
    }

    // 데이터 병합 함수
    public String[] mergeRows(String[] linear, String[] gyro, String[] gravity) {
        return new String[]{
                linear[1],                  // Timestamp
                linear[2], linear[3], linear[4], // Linear X, Y, Z
                gyro[2], gyro[3], gyro[4],       // Gyro X, Y, Z
                gravity[2], gravity[3], gravity[4] // Gravity X, Y, Z
        };
    }

    // Helper methods
    private int getColumnIndex(String column) {
        switch (column) {
            case "activity_class":
                return 0;
            case "timestamp":
                return 1;
            case "x":
                return 2;
            case "y":
                return 3;
            case "z":
                return 4;
            default:
                return -1;
        }
    }

    private double getQuantile(List<Double> data, double quantile) {
        List<Double> sorted = data.stream().filter(v -> !Double.isNaN(v)).sorted().collect(Collectors.toList());
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private void interpolate(List<Double> data) {
        int n = data.size();
        // Forward fill
        for (int i = 1; i < n; i++) {
            if (Double.isNaN(data.get(i)) && !Double.isNaN(data.get(i - 1))) {
                data.set(i, data.get(i - 1));
            }
        }
        // Backward fill
        for (int i = n - 2; i >= 0; i--) {
            if (Double.isNaN(data.get(i)) && !Double.isNaN(data.get(i + 1))) {
                data.set(i, data.get(i + 1));
            }
        }
        // Linear interpolation
        int start = -1;
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(data.get(i))) {
                if (start == -1) {
                    start = i;
                } else {
                    int end = i;
                    double startVal = data.get(start);
                    double endVal = data.get(end);
                    int gap = end - start;
                    for (int j = 1; j < gap; j++) {
                        data.set(start + j, startVal + (endVal - startVal) * j / gap);
                    }
                    start = end;
                }
            }
        }
    }
}



