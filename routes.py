from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, request, flash,  jsonify
from flask_mysqldb import MySQL, MySQLdb
import bcrypt
import werkzeug
import tensorflow as tf
from scipy import misc
from cv2 import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from skimage.transform import resize
import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re

app = Flask(__name__)

app.secret_key = "bigtuing"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'bigProject'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config['UPLOAD_FOLDER'] = 'pre_img/'

PATH = '\\'.join(os.path.abspath(__file__).split('\\')[0:-1])
DATASET_PATH = os.path.join(PATH, "pre_img")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/admin')
def admin():
    return render_template("login.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM admin WHERE username=%s", (username,) )
        user = curl.fetchone()
        curl.close()

        if user is not None and len(user) > 0:
            if password == user['password']:
                session['nama'] = user['nama']
                session['username'] = user['username']
                return redirect(url_for('dashboard'))
            else:
                flash("Gagal, username dan password tidak cocok")
                return redirect(url_for('login'))
        else:
            flash("Gagal, user tidak ditemukan")
            return redirect(url_for('login'))
    else:
        return render_template('login.html')

@app.route('/tamu')
def tamu():
    return render_template("formTamu.html")

@app.route('/modal')
def modal():
    return render_template("modal.html")

@app.route('/isiTamu', methods=['POST'])
def isiTamu():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        nama_lengkap = request.form['nama_lengkap']
        instansi = request.form['instansi']
        no_telp = request.form['no_telp']
        keperluan = request.form['keperluan']
        if not re.match(r'[A-Za-z]+', nama_lengkap):
            flash("Nama harus pakai huruf Dong!")
        elif not re.match(r'[0-9]+', no_telp):
            flash("No.Telepon harus pakai angka Dong!")
        
        else:
            cur.execute("INSERT INTO daftarTamu (nama_lengkap,instansi, no_telp, keperluan) VALUES (%s,%s,%s,%s)", (nama_lengkap,instansi, no_telp, keperluan))
            mysql.connection.commit()
            return render_template("modal.html")
            
    return render_template("formTamu.html")      


@app.route("/face_registration")
def face_registration():
    return render_template("face_registration.html")

@app.route("/uploadFoto", methods=['POST'])
def uploadFoto():
    class_name = request.args.get('class_name')
    path_new_class = os.path.join(DATASET_PATH, class_name)

    # create directory label if not exist
    if not os.path.exists(path_new_class):
        os.mkdir(path_new_class) 

    # save uploaded image
    filename = class_name + '%04d.jpg' % (len(os.listdir(path_new_class)) + 1) 
    file = request.files['webcam']
    file.save(os.path.join(path_new_class, filename))

    # resize
    img = cv2.imread(os.path.join(path_new_class, filename))
    img = cv2.resize(img, (250, 250))
    cv2.imwrite(os.path.join(path_new_class, filename), img)

    return '', 200



@app.route('/daftarKaryawan', methods=["POST", "GET"])
def daftarKaryawan():
    # karyawan = request.form.get("karyawan")
    if request.method == "POST":
        # karyawan = karyawan.strip().capitalize()
        # user_folder = os.path.join(app.config['UPLOAD_FOLDER'], karyawan)
        # os.mkdir(user_folder)
        # return f"folder is created under the name {karyawan} and the full path is {user_folder}"
        nama_karyawan = request.form['karyawan']
        email = request.form['email']
        no_telp = request.form['no_telp']
        alamat = request.form['alamat']
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("INSERT INTO karyawan (nama_lengkap,email, no_telp, alamat) VALUES (%s,%s,%s,%s)", (nama_karyawan,email, no_telp, alamat))
        mysql.connection.commit()
        flash("Karyawan berhasil ditambahkan")
        return render_template('karyawan.html')

@app.route('/karyawan')
def karyawan():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
 
    cur.execute('SELECT * FROM karyawan')
    data = cur.fetchall()
  
    cur.close()
    return render_template('karyawan.html', karyawan = data)

@app.route('/tambahData')
def tambahData():
    return render_template('tambahData.html')

@app.route('/rekamwajah')
def rekamwajah():
    return render_template("rekam.html")

pth = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(pth)

camera = cv2.VideoCapture(0)

@app.route('/regen')
def regen():
    count = 0
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            count = count + 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("pre_img/Don/User." + str(count) + ".jpg", gray[y:y+h,x:x+w])

            ret, buffer = cv2.imencode('.jpg', frame)

            k = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + k + b'\r\n')

            if k == 27:
                break
            elif count >= 30:
                
                # import webbrowser
                # webbrowser.open('http://127.0.0.1:5000/tangkap')
                break

@app.route('/tangkap')
def tangkap():
    return render_template("poptambah.html")

@app.route('/rekam')
def rekam():
    return Response(regen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tamuHari')
def tamuHari():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT COUNT(*) FROM daftarTamu WHERE tanggal = CURDATE()')
    total_hari_ini =  [v for v in cur.fetchone().values()][0]
    return total_hari_ini

@app.route('/tamuMinggu')
def tamuMinggu():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT COUNT(*) FROM daftarTamu GROUP BY YEARWEEK(tanggal);')
    total_minggu_ini =  [v for v in cur.fetchone().values()][0]
    return total_minggu_ini


@app.route('/tamuBulan')
def tamuBulan():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT COUNT(*) FROM daftarTamu WHERE MONTH(NOW());')
    total_bulan_ini =  [v for v in cur.fetchone().values()][0]

    return total_bulan_ini

@app.route('/totalTamu')
def totalTamu():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT COUNT(*) FROM daftarTamu')
    total_tamu =  [v for v in cur.fetchone().values()][0]
    return total_tamu

@app.route('/dashboard')
def dashboard():
    total_hari_ini = tamuHari()
    total_minggu_ini = tamuMinggu()
    total_bulan_ini = tamuBulan()
    total_tamu = totalTamu()
    return render_template('dashboard.html', total_hari_ini=total_hari_ini, total_minggu_ini=total_minggu_ini, total_bulan_ini=total_bulan_ini, total_tamu=total_tamu)

@app.route('/tabelTamu',methods=["POST", "GET"])
def tabelTamu():
    cursor = mysql.connection.cursor()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM daftarTamu ORDER BY id desc")
    totalTamu = cur.fetchall()
    return render_template('dataTamu.html', totalTamu=totalTamu)

@app.route('/warayah')
def warayah():
    return render_template('warayah.html')

@app.route("/ajaxfile",methods=["POST","GET"])
def ajaxfile():
    try:
        conn = mysql.connection.cursor()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        if request.method == 'POST':
            draw = request.form['draw'] 
            row = int(request.form['start'])
            rowperpage = int(request.form['length'])
            searchValue = request.form["search[value]"]
            print(draw)
            print(row)
            print(rowperpage)
            print(searchValue)
 
            ## Total number of records without filtering
            cursor.execute("select count(*) as allcount from daftarTamu")
            rsallcount = cursor.fetchone()
            totalRecords = rsallcount['allcount']
            print(totalRecords) 
 
            ## Total number of records with filtering
            likeString = "%" + searchValue +"%"
            cursor.execute("SELECT count(*) as allcount from daftarTamu WHERE nama_lengkap LIKE %s OR instansi LIKE %s OR no_telp LIKE %s", (likeString, likeString, likeString))
            rsallcount = cursor.fetchone()
            totalRecordwithFilter = rsallcount['allcount']
            print(totalRecordwithFilter) 
 
            ## Fetch records
            if searchValue=='':
                cursor.execute("SELECT * FROM daftarTamu ORDER BY nama_lengkap asc limit %s, %s;", (row, rowperpage))
                employeelist = cursor.fetchall()
            else:        
                cursor.execute("SELECT * FROM daftarTamu WHERE nama_lengkap LIKE %s OR instansi LIKE %s OR no_telp LIKE %s limit %s, %s;", (likeString, likeString, likeString, row, rowperpage))
                employeelist = cursor.fetchall()
 
            data = []
            for row in employeelist:
                data.append({
                    'tanggal': row['tanggal'],
                    'nama_lengkap': row['nama_lengkap'],
                    'instansi': row['instansi'],
                    'no_telp': row['no_telp'],
                    'keperluan': row['keperluan'],
                })
 
            response = {
                'draw': draw,
                'iTotalRecords': totalRecords,
                'iTotalDisplayRecords': totalRecordwithFilter,
                'aaData': data,
            }
            return jsonify(response)
    except Exception as e:
        print(e)
    finally:
        cursor.close() 
        conn.close()

@app.route("/range",methods=["POST","GET"])
def range(): 
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
    if request.method == 'POST':
        From = request.form['From']
        to = request.form['to']
        print(From)
        print(to)
        query = "SELECT * from daftarTamu WHERE tanggal BETWEEN '{}' AND '{}'".format(From,to)
        cur.execute(query)
        tgl = cur.fetchall()
    return jsonify({'htmlresponse': render_template('responDataTamu.html', tgl=tgl)})


@app.route('/hapusTamu/<string:id>', methods = ['POST','GET'])
def hapusTamu(id):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    cur.execute('DELETE FROM daftarTamu WHERE id = {0}'.format(id))
    mysql.connection.commit()
    flash('Pesan Masuk Berhasil Dihapus!')
    return redirect(url_for('tabelTamu'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

    
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./pre_img"

@app.route('/gen_frames')
def gen_frames():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        # HumanNames = "Karyawan"
        HumanNames.sort()

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(0)
        c = 0
        video_capture.set(3, 700) # set video width
        video_capture.set(4, 500) # set video height

        print('Start Recognition')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=1, fy=1 )    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(resize(cropped[i], output_shape=(image_size, image_size)))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")
                        print(best_class_indices,' with accuracy ',best_class_probabilities)
                        
                        # print(best_class_probabilities)
                        if best_class_probabilities>0.6:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    cv2.putText(frame, HumanNames, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                            # video_capture.release()
                            # cv2.destroyAllWindows()
                        else:
                            tamu = 'Tamu'
                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2) 
                            cv2.putText(frame, tamu, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)                            

                            # for tamu in range(1):
                            #     break
                            print('Anda Tamu')
                            # video_capture.release()
                            # cv2.destroyAllWindows()
                            # import webbrowser
                            # webbrowser.open_new('http://127.0.0.1:5000/tamu')

            ret, buffer = cv2.imencode('.jpg', frame)
            frames = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')      

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

model = load_model("chatbot\chatbot_model.h5")
intents = json.loads(open("chatbot\intents.json").read())
words = pickle.load(open("chatbot\words.pkl", "rb"))
classes = pickle.load(open("chatbot\classes.pkl", "rb"))


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == '__main__':
    app.run(debug=True)

