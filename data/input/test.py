import pickle

def check_pkl_format(pkl_path, show_sample=True, n_sample=3):
    print(f"---- Kiểm tra file: {pkl_path} ----")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print("Kiểu dữ liệu gốc:", type(data))

    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        # In thử value của key đầu
        k0 = next(iter(data))
        print(f"Value đầu: {type(data[k0])}")
    elif isinstance(data, list):
        print(f"List tổng số mẫu: {len(data)}")
        if show_sample:
            for i, item in enumerate(data[:n_sample]):
                print(f"--- Sample {i} ---")
                print("Kiểu:", type(item))
                if isinstance(item, dict):
                    print("Keys:", list(item.keys()))
                print(item if n_sample==1 else {k: item[k] for k in list(item)[:5]})
    else:
        # Có thể là DataFrame hoặc kiểu khác
        print("Cột (nếu là DataFrame):", getattr(data, 'columns', 'Không có'))
        if show_sample:
            try:
                print(data.head(n_sample))
            except Exception:
                print("Không thể hiển thị mẫu.")

if __name__ == "__main__":
    # Đổi tên thành file pkl muốn kiểm tra
    pkl_file = "0_cpg_input.pkl"
    check_pkl_format(pkl_file)
