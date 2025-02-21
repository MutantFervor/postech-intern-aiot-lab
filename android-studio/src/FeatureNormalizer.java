package com.example.activity_sensor_testing;

import android.content.Context;
import android.content.res.AssetManager;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvValidationException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class FeatureNormalizer {
    private final Context context;

    public FeatureNormalizer(Context context) {
        this.context = context;
    }

    public double[] normalizeFeatures(double[] features) {
        double[] normalizedFeatures = new double[features.length];

        try {
            // AssetManager를 통해 normalize_params.csv 파일 열기
            AssetManager assetManager = context.getAssets();
            InputStream inputStream = assetManager.open("normalize_params.csv");
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            CSVReader csvReader = new CSVReaderBuilder(bufferedReader).build();

            // Read parameters from normalize_params.csv
            String[] nextLine;
            int featureIndex = 0;

            // Skip the header (if applicable)
            csvReader.readNext();

            while ((nextLine = csvReader.readNext()) != null && featureIndex < features.length) {
                // Normalize the feature using a_i and b_i from params.csv
                double a_i = Double.parseDouble(nextLine[1]); // a_i 값
                double b_i = Double.parseDouble(nextLine[2]); // b_i 값
                normalizedFeatures[featureIndex] = a_i * features[featureIndex] + b_i;
                featureIndex++;
            }

            csvReader.close();
        } catch (IOException | CsvValidationException e) {
            e.printStackTrace();
            System.err.println("Error reading the normalize_params.csv file.");
            return null;
        }

        System.out.println("Features normalized successfully.");
        return normalizedFeatures;
    }
}
