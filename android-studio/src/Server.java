package com.example.activity_sensor_testing;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.widget.TextView;

import okhttp3.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Server {
    private final String serverUrl;
    private final OkHttpClient client;
    private final Handler mainHandler;

    public Server(String serverUrl) {
        this.serverUrl = serverUrl;
        this.client = new OkHttpClient();
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    // POST 메서드: 서버에 activity 데이터를 전송
    public void sendActivity(String activity, ServerCallback callback) {
        new Thread(() -> {
            try {
                // POST 요청 생성
                RequestBody requestBody = new FormBody.Builder()
                        .add("activity", activity)
                        .build();

                Request request = new Request.Builder()
                        .url(serverUrl + "/post_activity")
                        .post(requestBody)
                        .build();

                // 요청 실행
                Response response = client.newCall(request).execute();

                if (response.isSuccessful()) {
                    callback.onSuccess("Activity sent successfully: " + activity);
                } else {
                    callback.onFailure("Failed to send activity. Code: " + response.code());
                }
            } catch (Exception e) {
                callback.onFailure("Error: " + e.getMessage());
            }
        }).start();
    }

    // GET 메서드: 서버에서 실시간 스트리밍 데이터 수신
    public void startStreaming(TextView resultTextView) {
        Request request = new Request.Builder()
                .url(serverUrl + "/stream")
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                mainHandler.post(() -> resultTextView.setText("Streaming Error: " + e.getMessage()));
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(response.body().byteStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.startsWith("data: ")) {
                            String activity = line.substring(6).trim();
                            mainHandler.post(() -> resultTextView.setText("Received: " + activity));
                        }
                    }
                } else {
                    mainHandler.post(() -> resultTextView.setText("Stream failed. Code: " + response.code()));
                }
            }
        });
    }

    // Callback 인터페이스
    public interface ServerCallback {
        void onSuccess(String message);

        void onFailure(String error);
    }
}
