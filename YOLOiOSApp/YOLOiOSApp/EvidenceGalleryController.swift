import ImageIO
import Photos
import UIKit

/// Evidence stays in private app storage until the user explicitly exports an item.
final class EvidenceGalleryController: UICollectionViewController, UICollectionViewDelegateFlowLayout {
  private let store: AerialEvidenceStore
  private var items: [AerialEvidenceStore.Item] = []
  private let imageQueue = DispatchQueue(label: "evidence.thumbnails", qos: .userInitiated)

  init(store: AerialEvidenceStore) {
    self.store = store
    let layout = UICollectionViewFlowLayout()
    layout.minimumInteritemSpacing = 12
    layout.minimumLineSpacing = 16
    layout.sectionInset = UIEdgeInsets(top: 16, left: 16, bottom: 24, right: 16)
    super.init(collectionViewLayout: layout)
    title = "채증 갤러리"
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override func viewDidLoad() {
    super.viewDidLoad()
    collectionView.backgroundColor = .systemBackground
    collectionView.register(EvidenceCell.self, forCellWithReuseIdentifier: "evidence")
    navigationItem.leftBarButtonItem = UIBarButtonItem(barButtonSystemItem: .done,
      target: self, action: #selector(close))
    navigationItem.rightBarButtonItem = UIBarButtonItem(title: "보관 안내", style: .plain,
      target: self, action: #selector(storageInfo))
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    store.list { [weak self] result in
      guard let self else { return }
      switch result {
      case .success(let items):
        self.items = items
        let empty = UILabel()
        empty.text = "보관된 채증이 없습니다.\n자동 채증을 켜면 인식한 장면을 여기에 보관합니다."
        empty.textAlignment = .center; empty.numberOfLines = 0; empty.textColor = .secondaryLabel
        collectionView.backgroundView = items.isEmpty ? empty : nil
        collectionView.reloadData()
      case .failure(let error): message(error.localizedDescription)
      }
    }
  }

  @objc private func close() { dismiss(animated: true) }

  @objc private func storageInfo() {
    message("자동 채증은 사진·시각·분류 점수를 이 앱에만 보관합니다. 사진 앱에는 자동 저장하지 않습니다.\n\n같은 추적 대상은 최소 30초 간격, 전체 채증은 최소 5초 간격입니다. 최대 500MB에 도달하면 자동 채증을 끕니다. 항목을 열어 사진 앱에 저장하거나 공유·삭제할 수 있습니다.\n\n갤러리를 보는 동안은 새 채증을 잠시 멈춥니다. 앱을 삭제하면 보관 자료도 삭제되므로 필요한 자료는 먼저 내보내세요. 분류는 모델의 판단이며 실제 신원을 증명하지 않습니다.")
  }

  private func message(_ text: String) {
    let alert = UIAlertController(title: "채증 갤러리", message: text, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "확인", style: .default))
    present(alert, animated: true)
  }

  override func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
    items.count
  }

  override func collectionView(_ collectionView: UICollectionView,
                               cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "evidence", for: indexPath) as! EvidenceCell
    let item = items[indexPath.item]
    cell.itemID = item.id
    cell.picture.image = nil
    cell.caption.text = "\(item.summary)\n\(item.date.formatted(date: .abbreviated, time: .shortened))"
    cell.accessibilityLabel = cell.caption.text
    imageQueue.async {
      let image = Self.thumbnail(item.image, maxPixel: 480)
      DispatchQueue.main.async { [weak cell] in
        if cell?.itemID == item.id { cell?.picture.image = image }
      }
    }
    return cell
  }

  override func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
    navigationController?.pushViewController(EvidenceDetailController(item: items[indexPath.item], store: store),
      animated: true)
  }

  func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout,
                      sizeForItemAt indexPath: IndexPath) -> CGSize {
    let width = floor((collectionView.bounds.width - 44) / 2)
    return CGSize(width: width, height: width * 0.75 + 58)
  }

  nonisolated static func thumbnail(_ url: URL, maxPixel: Int) -> UIImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        kCGImageSourceShouldCacheImmediately: true
      ] as CFDictionary) else { return nil }
    return UIImage(cgImage: image)
  }
}

private final class EvidenceCell: UICollectionViewCell {
  let picture = UIImageView()
  let caption = UILabel()
  var itemID: String?

  override init(frame: CGRect) {
    super.init(frame: frame)
    picture.contentMode = .scaleAspectFit
    picture.backgroundColor = .secondarySystemBackground
    picture.clipsToBounds = true
    picture.layer.cornerRadius = 8
    caption.font = .systemFont(ofSize: 13)
    caption.numberOfLines = 2
    let stack = UIStackView(arrangedSubviews: [picture, caption])
    stack.axis = .vertical; stack.spacing = 6
    stack.translatesAutoresizingMaskIntoConstraints = false
    contentView.addSubview(stack)
    NSLayoutConstraint.activate([
      stack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
      stack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
      stack.topAnchor.constraint(equalTo: contentView.topAnchor),
      stack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
      caption.heightAnchor.constraint(equalToConstant: 48)
    ])
    isAccessibilityElement = true
  }

  override func prepareForReuse() {
    super.prepareForReuse()
    itemID = nil; picture.image = nil; caption.text = nil
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}

private final class EvidenceDetailController: UIViewController {
  private let item: AerialEvidenceStore.Item
  private let store: AerialEvidenceStore
  private let picture = UIImageView()
  private var exporting = false

  init(item: AerialEvidenceStore.Item, store: AerialEvidenceStore) {
    self.item = item; self.store = store
    super.init(nibName: nil, bundle: nil)
    title = item.summary
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .systemBackground
    picture.contentMode = .scaleAspectFit
    let caption = UILabel()
    caption.text = "\(item.date.formatted(date: .abbreviated, time: .standard))\n모델 판단이 표시된 사진입니다."
    caption.numberOfLines = 0; caption.textAlignment = .center
    let save = UIButton(type: .system)
    save.setTitle("사진 앱에 저장", for: .normal)
    save.addTarget(self, action: #selector(savePhoto), for: .touchUpInside)
    let share = UIButton(type: .system)
    share.setTitle("사진·채증 정보 내보내기", for: .normal)
    share.addTarget(self, action: #selector(shareFiles), for: .touchUpInside)
    let stack = UIStackView(arrangedSubviews: [picture, caption, save, share])
    stack.axis = .vertical; stack.spacing = 16; stack.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(stack)
    NSLayoutConstraint.activate([
      stack.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
      stack.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
      stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
      stack.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
      save.heightAnchor.constraint(equalToConstant: 44), share.heightAnchor.constraint(equalToConstant: 44)
    ])
    navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .trash,
      target: self, action: #selector(deleteItem))
    DispatchQueue.global(qos: .userInitiated).async { [item] in
      let image = EvidenceGalleryController.thumbnail(item.image, maxPixel: 2048)
      DispatchQueue.main.async { [weak self] in self?.picture.image = image }
    }
  }

  @objc private func savePhoto() {
    guard !exporting else { return }
    exporting = true
    let url = item.image
    // Add-only permission is requested only after this explicit user action.
    PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
      guard status == .authorized || status == .limited else {
        DispatchQueue.main.async {
          self?.exporting = false
          self?.message("저장 권한이 없어 사진 앱에 저장하지 않았습니다. 아이폰 설정에서 사진 추가 권한을 허용해 주세요.")
        }
        return
      }
      PHPhotoLibrary.shared().performChanges({
        PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: url)
      }) { success, error in
        DispatchQueue.main.async {
          self?.exporting = false
          self?.message(success ? "사진 앱에 저장했습니다. 앱 내 채증도 그대로 보관합니다."
            : (error?.localizedDescription ?? "사진 앱에 저장하지 못했습니다."))
        }
      }
    }
  }

  @objc private func shareFiles() {
    let share = UIActivityViewController(activityItems: [item.original, item.image, item.metadata],
      applicationActivities: nil)
    share.popoverPresentationController?.sourceView = view
    share.popoverPresentationController?.sourceRect = CGRect(x: view.bounds.midX, y: view.bounds.maxY - 60,
      width: 1, height: 1)
    present(share, animated: true)
  }

  @objc private func deleteItem() {
    guard !exporting else { return }
    let confirm = UIAlertController(title: "이 채증을 삭제할까요?",
      message: "앱 내 사진과 채증 정보가 삭제됩니다. 이미 사진 앱에 저장하거나 내보낸 자료는 유지됩니다.",
      preferredStyle: .alert)
    confirm.addAction(UIAlertAction(title: "취소", style: .cancel))
    confirm.addAction(UIAlertAction(title: "삭제", style: .destructive) { [weak self] _ in
      guard let self else { return }
      store.delete(item) { [weak self] result in
        switch result {
        case .success: self?.navigationController?.popViewController(animated: true)
        case .failure(let error): self?.message(error.localizedDescription)
        }
      }
    })
    present(confirm, animated: true)
  }

  private func message(_ text: String) {
    let alert = UIAlertController(title: "채증", message: text, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "확인", style: .default))
    present(alert, animated: true)
  }
}

