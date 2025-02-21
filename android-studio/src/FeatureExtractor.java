package com.example.activity_sensor_testing;
import java.util.*;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.TransformType;
import java.util.ArrayList;
import java.util.List;

public class FeatureExtractor {
    public static double[] extractFeatures(List<CombinedSensorData> windowData) {
        // 센서 데이터 분리
        double[] linearX = new double[windowData.size()];
        double[] linearY = new double[windowData.size()];
        double[] linearZ = new double[windowData.size()];
        double[] gyroX = new double[windowData.size()];
        double[] gyroY = new double[windowData.size()];
        double[] gyroZ = new double[windowData.size()];
        double[] gravityX = new double[windowData.size()];
        double[] gravityY = new double[windowData.size()];
        double[] gravityZ = new double[windowData.size()];

        for (int i = 0; i < windowData.size(); i++) {
            CombinedSensorData data = windowData.get(i);
            linearX[i] = data.linear_x;
            linearY[i] = data.linear_y;
            linearZ[i] = data.linear_z;
            gyroX[i] = data.gyro_x;
            gyroY[i] = data.gyro_y;
            gyroZ[i] = data.gyro_z;
            gravityX[i] = data.gravity_x;
            gravityY[i] = data.gravity_y;
            gravityZ[i] = data.gravity_z;
        }

        // 각 센서 축에 대해 피처 계산
        List<double[]> features = new ArrayList<>();
        features.add(calculateFeatures(linearX, linearY, linearZ));
        features.add(calculateFeatures(gyroX, gyroY, gyroZ));
        features.add(calculateFeatures(gravityX, gravityY, gravityZ));

        // 모든 피처를 하나의 배열로 병합
        int totalLength = features.stream().mapToInt(arr -> arr.length).sum();
        double[] mergedFeatures = new double[totalLength];
        int index = 0;

        for (double[] featureArray : features) {
            System.arraycopy(featureArray, 0, mergedFeatures, index, featureArray.length);
            index += featureArray.length;
        }

        return mergedFeatures;
    }

    public static double[] calculateFeatures(double[] windowX, double[] windowY, double[] windowZ) {
        // 1. DC
        double dcX = calculateMean(windowX);
        double dcY = calculateMean(windowY);
        double dcZ = calculateMean(windowZ);

        // 2. Frequency-domain entropy
        double hX = calculateEntropy(windowX);
        double hY = calculateEntropy(windowY);
        double hZ = calculateEntropy(windowZ);

        // 3. Total Energy of Frequency Spectrum
        double eX = calculateEnergy(windowX);
        double eY = calculateEnergy(windowY);
        double eZ = calculateEnergy(windowZ);

        // 4. Correlation
        PearsonsCorrelation correlation = new PearsonsCorrelation();
        double rXY = correlation.correlation(windowX, windowY);
        double rYZ = correlation.correlation(windowY, windowZ);
        double rXZ = correlation.correlation(windowX, windowZ);

        // +1. SMA (Signal Magnitude Area)
        double sma = calculateSMA(windowX, windowY, windowZ);

        // +2. Standard Deviation
        double stdX = calculateStandardDeviation(windowX);
        double stdY = calculateStandardDeviation(windowY);
        double stdZ = calculateStandardDeviation(windowZ);

        // +3. Skewness
        double skewX = calculateSkewness(windowX);
        double skewY = calculateSkewness(windowY);
        double skewZ = calculateSkewness(windowZ);

        // +4. Median Absolute Deviation
        double madX = calculateMAD(windowX);
        double madY = calculateMAD(windowY);
        double madZ = calculateMAD(windowZ);

        // +5. Kurtosis
        double kurtosisX = calculateKurtosis(windowX);
        double kurtosisY = calculateKurtosis(windowY);
        double kurtosisZ = calculateKurtosis(windowZ);

        // +6. Interquartile Range
        double iqrX = calculateIQR(windowX);
        double iqrY = calculateIQR(windowY);
        double iqrZ = calculateIQR(windowZ);

        return new double[]{
                dcX, hX, eX, rXY, rXZ, stdX,
                dcY, hY, eY, rYZ, stdY,
                dcZ, hZ, eZ, stdZ
        };

        /*return new double[]{
                dcX, hX, eX, rXY, rXZ, stdX, skewX, madX, kurtosisX, iqrX, sma,
                dcY, hY, eY, rYZ, stdY, skewY, madY, kurtosisY, iqrY,
                dcZ, hZ, eZ, stdZ, skewZ, madZ, kurtosisZ, iqrZ
        };*/
    }

    private static double calculateMean(double[] window) {
        double sum = 0.0;
        for (double val : window) {
            sum += val;
        }
        return sum / window.length;
    }

    private static double calculateEntropy(double[] window) {
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        double[] paddedWindow = padToPowerOfTwo(window); // 패딩 추가
        double[] magnitude = calculateMagnitude(transformer.transform(paddedWindow, TransformType.FORWARD));
        magnitude[0] = 0; // Exclude DC component

        double sumSquares = 0.0;
        for (double mag : magnitude) {
            sumSquares += mag * mag;
        }

        double entropy = 0.0;
        for (double mag : magnitude) {
            double prob = (mag * mag) / sumSquares;
            prob = Math.max(prob, 1e-10); // Avoid log(0)
            entropy -= prob * Math.log10(prob);
        }
        return entropy;
    }

    private static double[] padToPowerOfTwo(double[] window) {
        int length = window.length;
        int targetLength = Integer.highestOneBit(length);
        if (targetLength < length) {
            targetLength *= 2;
        }

        double[] paddedWindow = new double[targetLength];
        System.arraycopy(window, 0, paddedWindow, 0, length); // 기존 데이터 복사
        return paddedWindow;
    }

    private static double calculateEnergy(double[] window) {
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        double[] paddedWindow = padToPowerOfTwo(window); // 패딩 추가
        double[] magnitude = calculateMagnitude(transformer.transform(paddedWindow, TransformType.FORWARD));
        magnitude[0] = 0; // Exclude DC component

        double energy = 0.0;
        for (double mag : magnitude) {
            energy += mag * mag;
        }
        return energy / window.length;
    }

    private static double calculateSMA(double[] x, double[] y, double[] z) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.abs(x[i]) + Math.abs(y[i]) + Math.abs(z[i]);
        }
        return sum / 3.0;
    }

    private static double calculateStandardDeviation(double[] window) {
        double mean = calculateMean(window);
        double sum = 0.0;
        for (double val : window) {
            sum += Math.pow(val - mean, 2);
        }
        return Math.sqrt(sum / window.length);
    }

    private static double calculateSkewness(double[] window) {
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        double[] paddedWindow = padToPowerOfTwo(window); // 패딩 추가
        double[] magnitude = calculateMagnitude(transformer.transform(paddedWindow, TransformType.FORWARD));
        double mean = calculateMean(magnitude);
        double std = calculateStandardDeviation(magnitude);

        double skewness = 0.0;
        for (double mag : magnitude) {
            skewness += Math.pow((mag - mean) / std, 3);
        }
        return skewness / magnitude.length;
    }

    private static double calculateMAD(double[] window) {
        double median = calculateMedian(window);
        double sum = 0.0;
        for (double val : window) {
            sum += Math.abs(val - median);
        }
        return sum / window.length;
    }

    private static double calculateMedian(double[] window) {
        double[] copy = window.clone();
        java.util.Arrays.sort(copy);
        int middle = copy.length / 2;
        if (copy.length % 2 == 0) {
            return (copy[middle - 1] + copy[middle]) / 2.0;
        } else {
            return copy[middle];
        }
    }

    private static double calculateKurtosis(double[] window) {
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        double[] paddedWindow = padToPowerOfTwo(window); // 패딩 추가
        double[] magnitude = calculateMagnitude(transformer.transform(paddedWindow, TransformType.FORWARD));
        double mean = calculateMean(magnitude);

        double moment2 = 0.0;
        double moment4 = 0.0;
        for (double mag : magnitude) {
            double centered = mag - mean;
            moment2 += Math.pow(centered, 2);
            moment4 += Math.pow(centered, 4);
        }
        moment2 /= magnitude.length;
        moment4 /= magnitude.length;
        return moment4 / Math.pow(moment2, 2);
    }

    private static double calculateIQR(double[] window) {
        double q75 = calculatePercentile(window, 75);
        double q25 = calculatePercentile(window, 25);
        return q75 - q25;
    }

    private static double calculatePercentile(double[] window, double percentile) {
        double[] copy = window.clone();
        java.util.Arrays.sort(copy);
        double index = percentile / 100.0 * (copy.length - 1);
        int intIndex = (int) Math.floor(index);
        double frac = index - intIndex;
        if (intIndex + 1 < copy.length) {
            return copy[intIndex] + frac * (copy[intIndex + 1] - copy[intIndex]);
        } else {
            return copy[intIndex];
        }
    }

    private static double[] calculateMagnitude(Complex[] fftResult) {
        double[] magnitude = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            magnitude[i] = fftResult[i].abs();
        }
        return magnitude;
    }
}
