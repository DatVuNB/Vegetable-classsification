package test.vtd.vegetable_classification;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import test.vtd.vegetable_classification.ml.BfsModel;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture, upload;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        anhxa();

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    // Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 2);
            }
        });
    }

    private void anhxa() {
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        upload = findViewById(R.id.button2);
    }

    public void classifyImage(Bitmap image) {
        try {
            // Tạo một instance của mô hình đã được huấn luyện với tên BfsModel
            BfsModel model = BfsModel.newInstance(getApplicationContext());

            // Chuẩn bị đầu vào cho mô hình TensorFlow Lite
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1 , 224, 224, 3}, DataType.FLOAT32);

            // Tạo một ByteBuffer với kích thước đủ để chứa hình ảnh đầu vào với 224x224x3 kênh màu (RGB)
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Chuyển đổi hình ảnh từ Bitmap thành mảng các giá trị pixel (intValues)
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            // Duyệt qua từng pixel của hình ảnh
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // Lấy giá trị RGB của pixel hiện tại

                    byteBuffer.putFloat(((val >> 16) & 0xFF) ); // Kênh màu đỏ
                    byteBuffer.putFloat(((val >> 8) & 0xFF) );  // Kênh màu xanh lá
                    byteBuffer.putFloat((val & 0xFF) );         // Kênh màu xanh dương
                }
            }

            // Nạp dữ liệu ảnh đã chuẩn hóa vào input của mô hình
            inputFeature0.loadBuffer(byteBuffer);

            // Chạy suy luận trên mô hình với dữ liệu đầu vào đã được chuẩn bị
            BfsModel.Outputs outputs = model.process(inputFeature0);

            // Lấy đầu ra từ mô hình, đó là xác suất dự đoán cho từng lớp
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Mảng chứa xác suất dự đoán cho mỗi lớp
            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;

            // Tìm lớp có xác suất cao nhất
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {
                    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
                    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
                    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
            };

            // Hiển thị tên lớp có xác suất cao nhất trên giao diện người dùng
            result.setText(classes[maxPos]);

            // Hiển thị xác suất dự đoán của từng lớp dưới dạng phần trăm trên giao diện người dùng
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < classes.length; i++) {
                s.append(String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100));
            }

            confidence.setText(s.toString());

            // Đóng mô hình sau khi sử dụng để giải phóng tài nguyên
            model.close();
        } catch (IOException e) {
            e.printStackTrace(); // Xử lý ngoại lệ nếu có lỗi xảy ra
        }
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 1) { // Xử lý kết quả từ camera
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else if (requestCode == 2) { // Xử lý kết quả từ thư viện ảnh
                try {
                    // Lấy URI của ảnh được chọn
                    Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                    int dimension = Math.min(image.getWidth(), image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                    imageView.setImageBitmap(image);
                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                    classifyImage(image);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
