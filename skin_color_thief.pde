import gab.opencv.*;
import processing.video.*;
import org.opencv.core.Core;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfInt;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import java.awt.geom.Point2D;
import java.awt.Rectangle;
import java.nio.*;
import java.util.List;

OpenCV opencv;
Capture video;
Rectangle[] hands = new Rectangle[0];

PImage src, filtered;

int minH = 145, maxH = 225, minS = 8, maxS = 60, minB = 140, maxB = 200;
int elementSize = 14, blurSize = 5;

void setup() {
  size(1280, 480);
  video = new Capture(this, width / 2, height);
  video.start();
  
  opencv = new OpenCV(this, video);
  
  colorMode(HSB);
  textSize(20);
  thread("makeImages");
}
void draw() {
  if(src != null) image(src, 0, 0);
  if(filtered != null) image(filtered, src.width, 0);
  
  noFill();
  stroke(0, 255, 255);
  strokeWeight(3);
  
  text("Hue is " + minH + " to " + maxH, 20, 20);
  text("Sat is " + minS + " to " + maxS, 20, 60);
  text("Bri is " + minB + " to " + maxB, 20, 100);
  if(key == 'h') {
    minH = (int)map(mouseY, 0, height, 0, 360);
    maxH = (int)map(mouseX, 0, width, 0, 360);
  } else if(key == 's') {
    minS = (int)map(mouseY, 0, height, 0, 255);
    maxS = (int)map(mouseX, 0, width, 0, 255);
  } else if(key == 'b') {
    minB = (int)map(mouseY, 0, height, 0, 255);
    maxB = (int)map(mouseX, 0, width, 0, 255);
  }
}

void makeImages() {
  while(true) {
    if(video.available()) video.read();
    opencv.loadImage(video);
    opencv.useColor(); // rgb
    Mat frame = toMat(opencv.getSnapshot()); // set src
    frame = ARGBToRGBA(frame);
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
    
    // ready up the mat
    Mat toFilt = toMat(opencv.getSnapshot());
    toFilt = ARGBToRGBA(toFilt);
    Imgproc.cvtColor(toFilt, toFilt, Imgproc.COLOR_RGBA2RGB);
    Imgproc.cvtColor(toFilt, toFilt, Imgproc.COLOR_RGB2HSV);
    
    // filtering
    Core.inRange(toFilt, new Scalar(minH, minS, minB), new Scalar(maxH, maxS, maxB), toFilt);
    Imgproc.medianBlur(toFilt, toFilt, blurSize);
    Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * elementSize + 1, 2 * elementSize + 1), new Point(elementSize, elementSize));
    Imgproc.dilate(toFilt, toFilt, element);
    
    // contouring
    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat toFilt2 = toFilt.clone();
    Imgproc.findContours(toFilt, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
    toFilt = toFilt2;
    
    int largestContour = 0;
    for(int i = 1; i < contours.size(); i++) {
      if(Imgproc.contourArea(contours.get(i)) > Imgproc.contourArea(contours.get(largestContour)))
        largestContour = i;
    }
    ArrayList<MatOfPoint> largestContours = new ArrayList<MatOfPoint>();
    if(largestContour < contours.size()) largestContours.add(contours.get(largestContour));
    Imgproc.drawContours(frame, largestContours, -1, new Scalar(255, 60, 60), 3);
    
    // convex hulling
    if (contours.size() > 0)
    {
        MatOfInt hull = new MatOfInt();
        MatOfPoint lcontour = contours.get(largestContour);
        Imgproc.convexHull(lcontour, hull, false);
        
        MatOfPoint mopOut = new MatOfPoint();
        mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);
        
        for(int i = 0; i < hull.size().height ; i++)
        {
          int index = (int)hull.get(i, 0)[0];
          double[] point = new double[] {
            lcontour.get(index, 0)[0], lcontour.get(index, 0)[1]
          };
          mopOut.put(i, 0, point);
        }           
        ArrayList<MatOfPoint> mopOuts = new ArrayList<MatOfPoint>();
        mopOuts.add(mopOut);
        
        Imgproc.drawContours(frame, mopOuts, -1, new Scalar(60, 60, 255), 3);
        
        MatOfInt4 convDef = new MatOfInt4();
        Imgproc.convexityDefects(lcontour, hull, convDef);
        
        List<Integer> cdList = convDef.toList();
        Point data[] = lcontour.toArray();
        
        for(int i = 0; i < cdList.size(); i += 4) {
          Point start = data[cdList.get(i)];
          Point end = data[cdList.get(i+1)];
          Point defect = data[cdList.get(i+2)];
          
          Core.circle(frame, start, 5, new Scalar(0, 100, 0), 2);
        }
    }
    
    // update modded src image
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2RGBA);
    src = toPImage(RGBAToARGB(frame));
    
    // send back the mat
    Imgproc.cvtColor(toFilt, toFilt, Imgproc.COLOR_GRAY2RGBA);
    toFilt = RGBAToARGB(toFilt);
    PImage t = toPImage(toFilt);
    opencv.loadImage(t);
    
    filtered = opencv.getSnapshot();
  }
}

// Convert PImage (ARGB) to Mat (CvType = CV_8UC4)
Mat toMat(PImage image) {
  int w = image.width;
  int h = image.height;
  
  Mat mat = new Mat(h, w, CvType.CV_8UC4);
  byte[] data8 = new byte[w*h*4];
  int[] data32 = new int[w*h];
  arrayCopy(image.pixels, data32);
  
  ByteBuffer bBuf = ByteBuffer.allocate(w*h*4);
  IntBuffer iBuf = bBuf.asIntBuffer();
  iBuf.put(data32);
  bBuf.get(data8);
  mat.put(0, 0, data8);
  
  return mat;
}

// Convert Mat (CvType=CV_8UC4) to PImage (ARGB)
PImage toPImage(Mat mat) {
  int w = mat.width();
  int h = mat.height();
  
  PImage image = createImage(w, h, ARGB);
  byte[] data8 = new byte[w*h*4];
  int[] data32 = new int[w*h];
  mat.get(0, 0, data8);
  ByteBuffer.wrap(data8).asIntBuffer().get(data32);
  arrayCopy(data32, image.pixels);
  
  return image;
}

Mat ARGBToRGBA(Mat in) {
  ArrayList<Mat> channels_argb = new ArrayList<Mat>();
  Core.split(in, channels_argb);
  Mat ret = new Mat();
  ArrayList<Mat> channels_rgba = new ArrayList<Mat>();
  channels_rgba.add(channels_argb.get(1));
  channels_rgba.add(channels_argb.get(2));
  channels_rgba.add(channels_argb.get(3));
  channels_rgba.add(channels_argb.get(0));
  Core.merge(channels_rgba, ret);
  return ret;
}

Mat RGBAToARGB(Mat in) {
  ArrayList<Mat> channels_rgba = new ArrayList<Mat>();
  Core.split(in, channels_rgba);
  Mat ret = new Mat();
  ArrayList<Mat> channels_argb = new ArrayList<Mat>();
  channels_argb.add(channels_rgba.get(3));
  channels_argb.add(channels_rgba.get(0));
  channels_argb.add(channels_rgba.get(1));
  channels_argb.add(channels_rgba.get(2));
  Core.merge(channels_argb, ret);
  return ret;
}