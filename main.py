import threading
from flask_socketio import SocketIO, emit

from flask import Flask, render_template, Response, request, redirect, url_for
import json
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options

# 初始化路径
from video_processing import capture_frames

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 初始化Flask API
app = Flask(__name__)
socketio = SocketIO(app)

# 用于跟踪视频处理线程
processing_threads = {}


def predict(opt):
    results = model(**vars(opt), stream=True)
    for result in results:
        if opt.save_txt:
            result_json = json.loads(result.tojson())
            yield json.dumps({'results': result_json})
        else:
            im0 = cv2.imencode('.jpg', result.plot())[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    video_url = request.form.get('url')
    if video_url:
        return redirect(url_for('process', url=video_url))
    return render_template('index.html')


raw_data = '/home/dujiafeng/wkcode/yolo/data/raw'

@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    global opt  # 确保在函数内部访问到全局的opt变量
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')

        if uploaded_file:
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == 'T' else False

    elif request.method == 'GET':
        opt.source, opt.save_txt = update_options(request)

    return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process')
def process():
    video_url = request.args.get('url')
    return render_template('process.html', video_url=video_url)


@socketio.on('connect')
def handle_connect():
    video_url = request.args.get('url')
    sid = request.sid
    if video_url:
        stop_event = threading.Event()
        processing_thread = threading.Thread(target=start_video_capture, args=(video_url, stop_event, sid))
        processing_threads[sid] = (processing_thread, stop_event)
        processing_thread.start()


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in processing_threads:
        _, stop_event = processing_threads.pop(sid)
        stop_event.set()


def send_results(results, sid):
    boxes_json = []
    print(results.boxes)
    if results.boxes:
        # 获取boxes数据
        boxes = results.boxes.numpy()
        boxes_list = boxes.data

        for box in boxes_list:
            boxes_json.append({
                'x1': float(box[0]),
                'y1': float(box[1]),
                'x2': float(box[2]),
                'y2': float(box[3]),
                '识别结果': results.names[box[5]]
            })
    # 取 boxes和names
    res = {
        "boxes": boxes_json
    }
    socketio.emit('frame', {'data': json.dumps(res)}, room=sid)


def start_video_capture(video_url, stop_event, sid):
    capture_frames(video_url, lambda results: send_results(results, sid), stop_event)


if __name__ == '__main__':
    # 输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--weights', type=str, default=ROOT / 'yolov8n.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='source directory for images or videos')
    parser.add_argument('--conf', '--conf-thres', type=float, default=0.25,
                        help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show', '--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt', '--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels', '--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det', '--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true',
                        help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true',
                        help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list,
                        help='filter results by class, i.e. classes=0, or classes=[0,2,3]')  # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # 显示使用的参数
    print_args(vars(opt))

    # 进行部署
    port = opt.port
    delattr(opt, 'port')

    # 为原始文件创建路径
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')

    # 载入模型
    model = YOLO(str(opt.model))

    # 运行
    app.run(host='0.0.0.0', port=port, debug=True)
