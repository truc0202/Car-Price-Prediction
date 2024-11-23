# 🚙 Car Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## 🏠 OVERVIEW
Bối cảnh:

Một công ty ô tô Trung Quốc Geely Auto muốn thâm nhập thị trường Mỹ bằng cách thành lập đơn vị sản xuất của họ ở đó và sản xuất ô tô để cạnh tranh với các đối tác Mỹ và Châu Âu.

Do vậy, họ đã ký hợp đồng với một công ty tư vấn ô tô để hiểu các yếu tố ảnh hưởng đến việc định giá ô tô tại thị trường Mỹ, vì những yếu tố đó có thể rất khác so với thị trường Trung Quốc.

Vai trò:Tôi là một nhà phân tích dữ liệu cho công ty tư vấn ô tô, vai trò của tôi:
- Hiểu rõ về việc những yếu tố ảnh hưởng đến việc định giá ô tô tại thị trường Mỹ.
- Tìm hiểu xem có thể đưa ra mô hình dự đoán giá xe tại thị trường Mỹ.

Mục tiêu: xây dựng mô hình hướng tới việc 
- Xác đinh yếu tố quan trọng ảnh hưởng tới giá xe.
- Phát triển mô hình dự đoán giá.
- Phân tích giá theo phân khúc thịt trường

## :open_file_folder:   DATA DICTIONARY

| **Biến**              | **Loại giá trị**   | **Ý nghĩa**                                                                 |
|-----------------------|--------------------|-----------------------------------------------------------------------------|
| car_ID               | int64             | Mã định danh duy nhất của xe.                                               |
| symboling            | int64             | Xếp hạng rủi ro bảo hiểm của xe (càng cao rủi ro càng lớn).                 |
| CarName              | object            | Tên xe hoặc thương hiệu xe (ví dụ: Toyota, Honda).                          |
| fueltype             | object            | Loại nhiên liệu xe sử dụng (diesel hoặc gas).                               |
| aspiration           | object            | Loại tăng áp khí nạp (std: không có, turbo: có tăng áp).                    |
| doornumber           | object            | Số lượng cửa của xe (two: 2 cửa, four: 4 cửa).                              |
| carbody              | object            | Kiểu dáng xe (sedan, hatchback, convertible, v.v.).                         |
| drivewheel           | object            | Hệ thống truyền động (fwd: cầu trước, rwd: cầu sau, 4wd: 4 bánh).           |
| enginelocation       | object            | Vị trí động cơ (front: phía trước, rear: phía sau).                         |
| wheelbase            | float64           | Chiều dài cơ sở (khoảng cách giữa tâm bánh trước và bánh sau).              |
| carlength            | float64           | Chiều dài tổng thể của xe (đơn vị: inch hoặc cm).                           |
| carwidth             | float64           | Chiều rộng của xe (đơn vị: inch hoặc cm).                                   |
| carheight            | float64           | Chiều cao của xe (đơn vị: inch hoặc cm).                                    |
| curbweight           | int64             | Trọng lượng rỗng của xe (trọng lượng xe không tải).                         |
| enginetype           | object            | Loại động cơ (ohc, dohc, rotary, v.v.).                                     |
| cylindernumber       | object            | Số lượng xi-lanh trong động cơ (ví dụ: four, six).                          |
| enginesize           | int64             | Dung tích động cơ (đơn vị: cc hoặc inch khối).                              |
| fuelsystem           | object            | Hệ thống cung cấp nhiên liệu (mpfi: phun xăng đa điểm, 2bbl: bộ chế hòa khí 2 họng). |
| boreratio            | float64           | Tỷ lệ đường kính piston trên hành trình piston.                             |
| stroke               | float64           | Hành trình piston (khoảng cách piston di chuyển).                           |
| compressionratio     | float64           | Tỷ số nén (độ nén của hỗn hợp không khí và nhiên liệu).                     |
| horsepower           | int64             | Công suất của xe (đơn vị: mã lực - HP).                                     |
| peakrpm              | int64             | Tốc độ vòng tua cao nhất của động cơ (đơn vị: vòng/phút).                   |
| citympg              | int64             | Mức tiêu thụ nhiên liệu trong thành phố (dặm trên gallon - mpg).           |
| highwaympg           | int64             | Mức tiêu thụ nhiên liệu trên đường cao tốc (dặm trên gallon - mpg).        |
| price                | float64           | Giá bán của xe (đơn vị: USD hoặc loại tiền tệ khác).                        |

## 📊ANALYZE
Xác định yếu tố ảnh hưởng đến giá xe
![image](https://github.com/user-attachments/assets/d91c8632-fb4e-4f89-827a-f5756065db8c)
Nhận định: Các cột dữ liệu ngoại trừ symboling, cylindernumber, enginetype và fuelsystem không cho thấy sự ảnh hưởng đáng kể đến giá xe.
Giả thuyết:
- Giá trị của ô tô càng cao, mức độ rủi ro (symboling) sẽ càng thấp.
- Giá xe có xu hướng tỷ lệ thuận với mức độ phổ biến của xe và giá thành của các linh kiện trên xe.
![image](https://github.com/user-attachments/assets/57fc6b19-55e1-498e-a5b6-1b8d26492c7a)
![image](https://github.com/user-attachments/assets/4c1416ad-24c3-4e3a-a5e6-68ef6091991d)

## 🤖MODEL 
| **Mô hình**             | **MSE** | **R2_Test** |
|--------------------------|---------|-------------|
| Linear Regression        | 0.14    | 0.85        |
| Decision Tree            | 0.13    | 0.86        |
| Random Forest            | 0.13    | 0.87        |
| KNeighbors Regressor     | 0.20    | 0.69        |

## 🧾SUGGESTION
Dự đoán giá xe ô tô bằng mô hình học máy: Giải pháp cho thị trường minh bạch và hiệu quả
- Thị trường xe ô tô thường xuyên biến động với nhiều mức giá khác nhau khiến người mua gặp khó khăn trong
việc lựa chọn và so sánh giá cả. Nhằm giải quyết vấn đề này, việc ứng dụng mô hình học máy vào việc dự đoán giá
xe ô tô mang lại giải pháp tiềm năng cho thị trường minh bạch và hiệu quả hơn.
- Thông qua việc thu thập dữ liệu chi tiết về thuộc tính xe từ nhiều nguồn uy tín, mô hình học máy được xây dựng
và huấn luyện để dự đoán giá xe chính xác. Mô hình này mang lại nhiều lợi ích thiết thực cho cả người mua xe và
đại lý xe:

**Đối với người mua xe:**

• Dễ dàng so sánh giá xe từ các nguồn khác nhau, tránh mua xe giá cao.

• Lựa chọn mua xe phù hợp với nhu cầu và ngân sách.

• Tiết kiệm thời gian và công sức trong quá trình tìm kiếm và mua xe.

**Đối với đại lý xe:**

• Định giá xe cạnh tranh và hợp lý hơn, thu hút khách hàng tiềm năng.

• Tăng doanh số bán hàng và lợi nhuận.

• Giảm chi phí hoạt động và quản lý.

Việc triển khai mô hình học máy dự đoán giá xe ô tô sẽ góp phần tạo dựng thị trường xe minh bạch, hiệu quả và
lành mạnh hơn, mang lại lợi ích cho cả người mua và người bán.
