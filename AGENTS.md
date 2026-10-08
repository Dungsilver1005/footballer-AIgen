# Hướng dẫn làm việc trong dự án

Tài liệu này hợp nhất các hướng dẫn trước đây trong `.agent/`. Đọc file này trước khi bắt đầu task; sau đó chỉ đọc phần tài liệu và source liên quan đến task đó.

## 1. Đây là project gì?

**Football Player Generative AI** là project local tạo ảnh cầu thủ bóng đá Việt Nam theo prompt, dùng Stable Diffusion 1.5 cùng LoRA. Repository có dataset ảnh và caption, cấu hình huấn luyện, model/checkpoint, script inference và output ảnh. Các phong cách anime, cinematic và action được mô tả trong README.

Các vị trí chính:

- `dataset/VNfootballer/`: ảnh `.jpg` và caption `.txt` dùng cho huấn luyện.
- `captioning_rules.md`, `controlled_caption_vocab.md`: quy tắc viết caption và từ vựng được kiểm soát.
- `configs/`: cấu hình huấn luyện; `configs/footballer_lora_512_ep3.toml` là cấu hình dự án hiện có.
- `models/`, `outputs/lora/`: model nền và LoRA/checkpoint.
- `inference_outputs/`: ảnh inference đã sinh.
- `tools/sd-scripts/`: bộ công cụ huấn luyện sd-scripts. Kiểm tra cấu hình và lệnh chạy trước khi kết luận bộ công cụ này đang được dùng cho một run cụ thể.
- `.docs/`: ghi chú kiến trúc, pipeline, cấu hình, bản đồ code và trạng thái dự án.

**Kiểm tra cây file thực tế trước khi dựa vào tài liệu cũ.** Hiện root có `inference_backup.py` với CLI và `main()`, nhưng không có `inference.py`; README và một số file `.docs/` vẫn nhắc tới `inference.py`. Hãy xác định script và lệnh chạy thực sự từ source/caller trước khi chạy, đổi tên hoặc sửa luồng inference. Thư mục `src/` hiện chưa có source file.

## 2. Phải tuân theo quy tắc nào?

- Với hành vi runtime, source code hiện tại là căn cứ chính. Với thông số huấn luyện, kiểm tra config/run artifact tương ứng. README và `.docs/` là tài liệu tham khảo và có thể đã cũ; sửa tài liệu nếu phát hiện chúng mâu thuẫn với source.
- Trước khi sửa: đọc task, xác định phạm vi và subsystem; đọc `.docs/PROJECT_OVERVIEW.md` cùng tài liệu subsystem phù hợp; kiểm tra source thật và tìm nơi gọi/sử dụng bằng `rg`.
- Chỉ sửa phần cần cho task. Không refactor lớn, đổi kiến trúc, xóa behavior, đổi model/dataset/tham số huấn luyện/tham số sinh ảnh hoặc thêm dependency nếu task không yêu cầu rõ.
- Ưu tiên abstraction và utility sẵn có; không tạo wrapper hay hệ thống trùng lặp không cần thiết. Giữ style hiện tại và tương thích ngược khi hợp lý.
- Trước khi xóa/đổi tên hàm hoặc đổi interface, tìm toàn bộ usages và kiểm tra callers. Không xóa legacy code khi chưa xác minh dependency.
- Dùng path tương đối theo project root hoặc cấu hình hiện hữu; tránh hardcode absolute path.
- Không xem model asset là tính năng đang hoạt động nếu chưa tìm thấy caller và execution path. Với training, phân biệt code trong `tools/sd-scripts/`, config/artifact của dự án và lệnh/run thực tế; không suy đoán command, preprocessing hay kết quả.
- Không mở rộng task để tự sửa vấn đề ngoài phạm vi. Ghi nhận và báo cáo vấn đề đó.
- Chỉ ghi trong tài liệu command, path, class/function và flow đã kiểm chứng. Nếu thiếu bằng chứng, ghi `UNKNOWN / NEEDS VERIFICATION`.

## 3. Làm task X thì đọc/chỉnh ở đâu?

| Loại task | Đọc trước | Vị trí cần kiểm tra/chỉnh |
|---|---|---|
| Hiểu tổng thể dự án hoặc tìm module | `.docs/PROJECT_OVERVIEW.md`, `.docs/CODEBASE_MAP.md`, README | Xác nhận lại bằng cây file và source hiện tại. |
| Inference, CLI, prompt/tham số sinh ảnh | `.docs/EXECUTION_FLOW.md`, `.docs/DEVELOPMENT_GUIDE.md`, `.docs/MODEL_PIPELINE.md` | Tìm script inference thực tế; hiện có `inference_backup.py`. Xem `build_parser()`, `main()` và lệnh gọi trước khi sửa. |
| Nạp model nền, LoRA, memory/offload | `.docs/MODEL_PIPELINE.md`, `.docs/ARCHITECTURE.md` | Trong script thực tế, kiểm tra `load_pipeline()`, `find_lora()`, `load_lora_weights_compatibly()` và `enable_memory_optimizations()` nếu các hàm còn tồn tại. Kiểm tra thêm `models/` và `outputs/lora/`. |
| Tên/nơi lưu ảnh sinh ra | `.docs/EXECUTION_FLOW.md`, `.docs/DEVELOPMENT_GUIDE.md` | Hằng số output và đoạn lưu ảnh trong script inference; output hiện lưu dưới `inference_outputs/`. |
| Dataset, caption, preprocessing | `.docs/DATA_PIPELINE.md`, `captioning_rules.md`, `controlled_caption_vocab.md` | `dataset/VNfootballer/` và config liên quan trong `configs/`. Giữ cặp ảnh-caption cùng basename; xác minh ảnh hưởng tới reader huấn luyện trước khi đổi format. |
| Tham số hoặc quy trình huấn luyện | `.docs/DATA_PIPELINE.md`, `.docs/CONFIG_AND_DEPENDENCIES.md`, `.docs/EXECUTION_FLOW.md` | `configs/`, run snapshots trong `outputs/lora/` và phần liên quan trong `tools/sd-scripts/`. Xác minh lệnh, phiên bản và reader thực tế; đừng coi snapshot là cấu hình đang được nạp tự động. |
| Dependency, môi trường, path | `.docs/CONFIG_AND_DEPENDENCIES.md`, `.docs/DEVELOPMENT_GUIDE.md` | Source/config hiện hành, `.gitignore` và manifest dependency thực sự liên quan; kiểm tra môi trường trước khi khẳng định lệnh chạy được. |
| Tài liệu hoặc trạng thái dự án | Bài `.docs/` liên quan và source/config bị ảnh hưởng | Cập nhật các tài liệu `.docs/` theo bảng bên dưới; README root dành cho người dùng/developer, không thay cho hướng dẫn agent này. |

### Khi nào cập nhật `.docs/`?

| Thay đổi | Tài liệu cần cập nhật |
|---|---|
| Kiến trúc model hoặc cách nạp model | `MODEL_PIPELINE.md`, `ARCHITECTURE.md` |
| Entry point hoặc luồng training/inference | `EXECUTION_FLOW.md`, `DEVELOPMENT_GUIDE.md` |
| Dataset, caption hoặc preprocessing | `DATA_PIPELINE.md` |
| Config, dependency, môi trường hoặc path | `CONFIG_AND_DEPENDENCIES.md` |
| Cấu trúc file/thư mục hoặc module quan trọng | `CODEBASE_MAP.md` |
| TODO, debt, trạng thái hoàn thành/kiểm chứng | `CURRENT_STATUS.md` |
| Cách hiểu tổng thể về project | `PROJECT_OVERVIEW.md` |

Không cần sửa `.docs/` cho typo, lỗi cục bộ nhỏ hoặc formatting không ảnh hưởng cách hiểu/chạy hệ thống. Nếu trạng thái chưa rõ, cập nhật `CURRENT_STATUS.md` hoặc ghi `UNKNOWN / NEEDS VERIFICATION`; không suy đoán. Không tạo `.docs/README.md`.

## 4. Kiểm tra và báo cáo thế nào?

Sau khi sửa:

1. Rà soát diff và xác nhận chỉ có file thuộc phạm vi task bị thay đổi.
2. Tìm lại usages/callers và kiểm tra path/interface liên quan nếu task tác động tới chúng.
3. Chạy kiểm tra phù hợp với thay đổi và môi trường: ví dụ kiểm tra cú pháp cho Python, kiểm tra CLI, test sẵn có, hoặc smoke inference khi model/dependency/phần cứng cần thiết khả dụng. Không tạo test suite ngoài phạm vi task.
4. Kiểm tra tài liệu/path/command được sửa có khớp source hiện hành; cập nhật `.docs/` nếu thay đổi làm ảnh hưởng cách hiểu hoặc cách chạy hệ thống.
5. Không ghi kiểm tra là “đạt” nếu chưa chạy. Nêu rõ kiểm tra bị bỏ qua hoặc bị chặn và lý do.

Báo cáo cuối task ngắn gọn, gồm:

- **Đã đổi:** file chính và nội dung thay đổi.
- **Đã kiểm tra:** lệnh/kiểm tra đã chạy và kết quả; nêu rõ mục chưa chạy cùng lý do.
- **Tài liệu:** `.docs/` nào đã cập nhật, hoặc vì sao không cần.
- **Còn cần xác minh:** vấn đề ngoài phạm vi hoặc thiếu bằng chứng, nếu có.
