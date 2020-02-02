package com.example.yolo;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.SurfaceView;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.physicaloid.lib.Physicaloid;
import com.physicaloid.lib.usb.driver.uart.ReadLisener;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, SensorEventListener {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Net tinyYolo;
    TextView angle,confidenceView,CxView,CyView,dirText;
    SensorManager sensorManager;
    Physicaloid mPhysicaloid;
    ImageView indicator;
    Button startBut,connBut;

    boolean startYolo = false, firstTimeYolo = false, arduinoButton = false, firstTimeOpen=true, firstTimeClose=false;
    int arduinoButtonCount=0, initialAngle=0;

    private AlphaAnimation buttonClick = new AlphaAnimation(2F, 0.2F);

    public void YOLO(View Button){
        Button.startAnimation(buttonClick);
        if (startYolo == false){
            startYolo = true;

            startBut.setText("STOP");
            if (firstTimeYolo == false)
            {
                firstTimeYolo = true;
                String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;
                //String tinyYoloCfg = Environment.
                String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";
                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
            }
        }
        else
            {
            startBut.setText("START");
            startYolo = false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        angle = findViewById(R.id.angle);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setMaxFrameSize(800,600);//1280-960
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        indicator = findViewById(R.id.indicator);
        startBut = findViewById(R.id.button3);
        connBut = findViewById(R.id.connBut);
        confidenceView =  findViewById(R.id.confidenceView);
        CxView = findViewById(R.id.Cx);
        CyView = findViewById(R.id.Cy);
        angle = findViewById(R.id.angle);
        dirText = findViewById(R.id.dirText);

        mPhysicaloid = new Physicaloid(this);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch(status){
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        System.out.println(frame.size());
        if (startYolo) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416),new Scalar(0, 0, 0),false, false);
            tinyYolo.setInput(imageBlob);
            List<Mat> result = new ArrayList<Mat>(2);

            List<String> outBlobNames = new ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYolo.forward(result,outBlobNames);

            float confThreshold = 0.5f;

            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();

            int centerX =0;
            int centerY=0;

            for (int i = 0; i < result.size(); ++i)
            {
                Mat level = result.get(i);
                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float)mm.maxVal;

                    Point classIdPoint = mm.maxLoc;
                    if (confidence > confThreshold)
                    {
                        centerX = (int)(row.get(0,0)[0] * frame.cols());
                        centerY = (int)(row.get(0,1)[0] * frame.rows());
                        int width   = (int)(row.get(0,2)[0] * frame.cols());
                        int height  = (int)(row.get(0,3)[0] * frame.rows());
                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;
                        clsIds.add((int)classIdPoint.x);
                        confs.add((float)confidence);
                        rects.add(new Rect(left, top, width, height));

                    }
                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                float nmsThresh = 0.2f;
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

                int[] ind = indices.toArray();
                int intConf;

               for (int i = 0; i < ind.length; ++i)
                {
                    int idx = ind[i];
                    Rect box = boxesArray[idx];
//                    int idGuy = clsIds.get(idx);
                    float conf = confs.get(idx);
//                    List<String> cocoNames = Arrays.asList("Rugby_ball");
                    intConf = (int) (conf * 100);

                    final int finalIntConf = intConf;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            confidenceView.setText(Integer.toString(finalIntConf)+"%");
                        }
                    });

                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 255, 0), 1);
                }

                CyView.setText(String.valueOf(centerY));
                CxView.setText(String.valueOf(centerX));

                Imgproc.line(frame,new Point(centerX,0),new Point(centerX,768),new Scalar(245,212,66),2);      //Horizontal line blue
                Imgproc.line(frame,new Point(0,centerY),new Point(1024,centerY),new Scalar(245,212,66),2);

                Imgproc.circle(frame,new Point(centerX,centerY),13,new Scalar(0,0,200),9);

                //vertical line
                if(centerX<frame.rows()*0.37 && centerX>frame.rows()*0.17&& centerY<frame.cols()*0.24 && centerY>frame.cols()*0.14)
                    dirText.setText("GOOD");
                else if(centerX>frame.rows()*0.37)
                    dirText.setText("RIGHT");
                else if(centerX<frame.rows()*0.17)
                    dirText.setText("LEFT");
                else if(centerY<frame.cols()*0.14)
                    dirText.setText("BACK");
                else if(centerY<frame.cols()*0.24)
                    dirText.setText("FORWARD");
            }
        }

        if(arduinoButtonCount%2 != 0 )
        {
            if(firstTimeOpen){
                firstTimeOpen=false;
                firstTimeClose=true;
                arduinoButton=true;
                startArduinoComm();} }
        else if(arduinoButtonCount!=0){
            if(firstTimeClose){
                firstTimeClose=false;
                firstTimeOpen=true;
                arduinoButton=false;
                stopArduinoComm();
            }
        }

        if(arduinoButton){
            //onWrite(direction);
            onWrite("x");
        }
        Imgproc.line(frame,new Point(frame.rows()*0.17,0),new Point(frame.rows()*0.17,768),new Scalar(255,0,0),2);      //Horizontal line blue
        Imgproc.line(frame,new Point(frame.rows()*0.37,0),new Point(frame.rows()*0.37,768),new Scalar(255,0,0),2);      //Horizontal line blue
        Imgproc.line(frame,new Point(0,frame.cols()*0.24),new Point(1024,frame.cols()*0.24),new Scalar(255,0,0),2);
        Imgproc.line(frame,new Point(0,frame.cols()*0.14),new Point(1024,frame.cols()*0.14),new Scalar(255,0,0),2);
        return frame;
    }

    public void onWrite(String directionStr) { 	    // when send button is prressed
        String str = directionStr +"\r\n";	        //the points that we have to send to arduino
        if(str.length()>0) {
            byte[] buf = str.getBytes();	        //convert string to byte array
            mPhysicaloid.write(buf, buf.length);	//write data to arduino
        }
    }

    public void startArduinoComm() {
        connBut.setText("DISCONNECT");
        indicator.setImageResource(R.drawable.green);
        mPhysicaloid.setBaudrate(9600);
        if(mPhysicaloid.open()) {
            mPhysicaloid.addReadListener(new ReadLisener() {
                @Override
                public void onRead(int size) {
                    byte[] buf = new byte[size];
                    mPhysicaloid.read(buf, size);
                }
            });
        }
    }

    public void stopArduinoComm() {
        connBut.setText("CONNECT");
        indicator.setImageResource(R.drawable.red);
        if(mPhysicaloid.close()) { 	             //close the connection to arduino
            mPhysicaloid.clearReadListener();	//clear read listener
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        if (startYolo == true)
        {
            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;
            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";
            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
        }
        }

        @Override
        public void onCameraViewStopped() {
        }

    @Override
    protected void onResume() {
        super.onResume();
        sensorManager.registerListener(this,sensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION),
                SensorManager.SENSOR_DELAY_GAME);
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
        sensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        int degree = Math.round(event.values[0]);
          angle.setText(Integer.toString(degree)+"\u00B0");
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    public void connect(View view) {
        view.startAnimation(buttonClick);
        arduinoButtonCount++;
    }
}