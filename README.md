*Khoá luận tốt nghiệp*
# Tóm tắt video dựa trên mô hình học không giám sát

Thành viên:
- Nguyễn Thành Luân, 19120285.
- Nguyễn Phùng Mai Đan, 19120466.

Giảng viên hướng dẫn:
- TS. Võ Hoài Việt.


# Câu lệnh sử dụng
## Tạo bộ dữ liệu mới

*Link tải video của hai bộ dữ liệu SumMe và TVSum*
- SumMe: https://gyglim.github.io/me/vsum/index.html
- TVSum: http://people.csail.mit.edu/yalesong/tvsum/

Đưa toàn bộ video của tập dữ liệu (SumMe hoặc TVSum) vào một thư mục (videos/) và sử dụng lệnh sau
```bash
python create_data.py --input videos/ --output datasets/my_dataset.h5 --extract-method his --bin 32
```
Trong đó '--input' là địa chỉ thư mục chứa video, '--output' là địa chỉ và tên file h5 kết quả muốn tạo, '--extract-method' là phương pháp trích xuất đặc trưng, '--bin' là số bin RBG khi sử dụng phương pháp Color Histogram để rút trích đặc trưng.
*Xem thêm chi tiết trong file 'create_data.py'*

## Thêm tóm tắt của người dùng vào bộ dữ liệu vừa tạo
Dữ liệu về video tóm tắt của người dùng được để trong thư mục user_summary_h5_file (SumMe và TVSum), để thêm vào bộ dữ liệu sử dụng lệnh sau
```bash
python add_user_summary.py --input datasets/my_dataset.h5 --output datasets/summe_his.h5 --dataset-name summe
```
Trong đó '--input' là địa chỉ file chứa bộ dữ liệu, '--output' là địa chỉ file bộ dữ liệu kết quả sau khi thêm, '--dataset-name' là tên bộ dữ liệu để thêm vào (phụ thuộc vào bộ video sử dụng ban đầu).
*Xem thêm chi tiết trong file 'add_user_summary.py'*

## Tạo file split để thực hiện quá trình training
Chạy lệnh sau để tạo file split
```bash
python create_split.py -d datasets/summe_his.h5 --save-dir datasets --save-name summe_splits  --num-splits 5
```
Trong đó, '-d' là địa chỉ file chứa bộ dữ liệu, '--save-dir' là địa chỉ thư mục chứa file split sau khi tạo, '--save-name' là tên file split, '--num-splits' là số lượng bộ split muốn tạo
*Xem thêm chi tiết trong file 'create_split.py'*

## Đào tạo mô hình
Chạy lệnh sau để đào tạo mô hình sau khi đã có bộ dữ liệu
```bash
python main.py -d datasets/summe_his.h5 -s datasets/summe_splits.json -vt summe --gpu 0 --split-id 0 --verbose --input-size 96
```
Trong đó, '-d' là địa chỉ file chứa bộ dữ liệu, '-s' là địa chỉ file split, '-vt' là loại video sử dụng trong bộ dữ liệu (SumMe hoặc TVSum), '--split-id' là thứ tự của bộ split muốn sử dụng, '--verbose' là xác nhận muốn in thông tin kết quả đào tạo, '--input-size' là kích thước của đặc trưng khung hình trích xuất.
*Xem thêm chi tiết trong file 'utils/configs.py'*

## Chạy thực nghiệm với video mới
Sau khi đã có mô hình được đào tạo, để tóm tắt một video mới, đưa video muốn tóm tắt vào một thư mục (input/) và chạy lệnh sau
```bash
python video_summary.py --input input/test.mp4 --model Summaries/summe/model_epoch60.pth.tar --extract-method his --bin 32 --input-size 96
```
Trong đó, '--input' là địa chỉ file video muốn tóm tắt, '--model' là địa chỉ mô hình đã được đào tạo, '--extract-method' là phương pháp trích xuất đặc trưng, '--bin' là số bin RBG khi sử dụng phương pháp Color Histogram để rút trích đặc trưng, '--input-size' là kích thước của đặc trưng khung hình trích xuất.
*Xem thêm chi tiết trong file 'video_summary.py'*


