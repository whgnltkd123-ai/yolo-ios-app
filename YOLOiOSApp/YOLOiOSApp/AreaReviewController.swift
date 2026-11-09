import UIKit
import UltralyticsYOLO

/// Reviews a single photo locally. No scan results are forwarded to the live alarm.
final class AreaReviewController: UIViewController, UIScrollViewDelegate {
  private let photo: UIImage
  private let scroll = UIScrollView()
  private let imageView = UIImageView()
  private let status = UILabel()
  private let reviewLocations = CALayer()
  private let closeAction: () -> Void
  private var layoutSize = CGSize.zero
  private var scanTask: Task<Void, Never>?

  init(photo: UIImage, onClose: @escaping () -> Void) {
    self.photo = photo
    closeAction = onClose
    super.init(nibName: nil, bundle: nil)
    title = "정밀 탐색"
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black
    navigationItem.leftBarButtonItem = UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(close))
    scroll.delegate = self
    scroll.translatesAutoresizingMaskIntoConstraints = false
    status.translatesAutoresizingMaskIntoConstraints = false
    status.font = .preferredFont(forTextStyle: .footnote)
    status.textColor = .white
    status.numberOfLines = 0
    status.text = "촬영한 사진의 네 구역을 탐색하고 있습니다."
    view.addSubview(scroll); view.addSubview(status)
    let safe = view.safeAreaLayoutGuide
    NSLayoutConstraint.activate([
      scroll.topAnchor.constraint(equalTo: safe.topAnchor), scroll.leadingAnchor.constraint(equalTo: safe.leadingAnchor),
      scroll.trailingAnchor.constraint(equalTo: safe.trailingAnchor), scroll.bottomAnchor.constraint(equalTo: status.topAnchor, constant: -6),
      status.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 12),
      status.trailingAnchor.constraint(equalTo: safe.trailingAnchor, constant: -12),
      status.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -8),
    ])
    imageView.image = photo
    imageView.frame = CGRect(origin: .zero, size: photo.size)
    scroll.addSubview(imageView)
    scroll.contentSize = photo.size
    guard let url = Bundle.main.url(forResource: "C7Area", withExtension: "mlmodelc", subdirectory: "Models/Area") else {
      status.text = "이 빌드에 정밀 탐색 모델이 없습니다."
      return
    }
    let input = photo
    scanTask = Task.detached(priority: .userInitiated) { [weak self] in
      do {
        let result = try AerialAreaSearch.inspect(input, modelURL: url)
        try Task.checkCancellation()
        await MainActor.run { [weak self] in self?.display(result) }
      } catch is CancellationError {
        return
      } catch {
        await MainActor.run { [weak self] in self?.status.text = error.localizedDescription }
      }
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    guard scroll.bounds.size != layoutSize, min(scroll.bounds.width,scroll.bounds.height)>0 else { return }
    layoutSize = scroll.bounds.size
    let fit = min(layoutSize.width/photo.size.width,layoutSize.height/photo.size.height)
    scroll.minimumZoomScale = fit
    scroll.maximumZoomScale = max(1,fit*8)
    scroll.setZoomScale(fit, animated: false)
    scrollViewDidZoom(scroll)
  }

  func viewForZooming(in scrollView: UIScrollView) -> UIView? { imageView }
  func scrollViewDidZoom(_ s: UIScrollView) {
    s.contentInset = UIEdgeInsets(top:max(0,(s.bounds.height-s.contentSize.height)/2),
      left:max(0,(s.bounds.width-s.contentSize.width)/2),bottom:0,right:0)
  }

  @objc private func close() { dismiss(animated:true) }
  override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)
    if navigationController?.isBeingDismissed == true || view.window == nil {
      scanTask?.cancel()
      closeAction()
    }
  }

  private func display(_ result: AerialAreaReview) {
    let size = imageView.bounds.size
    let factor = CGAffineTransform(scaleX: size.width / result.size.width, y: size.height / result.size.height)
    let font = UIFont.systemFont(ofSize: max(12, max(size.width, size.height) * 0.008), weight: .semibold)
    func draw(_ box: Box, _ text: String, _ color: UIColor, on parent: CALayer) {
      let rect = box.xywh.applying(factor)
      let outline = CALayer()
      outline.frame = rect
      outline.borderColor = color.cgColor
      outline.borderWidth = max(1.5, font.pointSize * 0.12)
      parent.addSublayer(outline)
      let label = CATextLayer()
      label.contentsScale = UIScreen.main.scale
      label.string = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: color])
      label.backgroundColor = UIColor.black.withAlphaComponent(0.75).cgColor
      let width = ceil((text as NSString).size(withAttributes: [.font: font]).width) + 2
      label.frame = CGRect(x: max(0, min(rect.minX, size.width-width)), y: max(0, rect.minY-font.lineHeight),
        width: width, height: ceil(font.lineHeight))
      parent.addSublayer(label)
    }
    for box in result.original {
      draw(box, box.aerialDisplayLabel ?? box.cls, .white, on: imageView.layer)
    }
    status.text = "기존 모델 판정 표시"
    guard !result.verified.isEmpty else { return }
    reviewLocations.frame = imageView.bounds
    reviewLocations.isHidden = true
    imageView.layer.addSublayer(reviewLocations)
    for box in result.verified { draw(box, "재검토 위치", .systemTeal, on: reviewLocations) }
    navigationItem.rightBarButtonItem = UIBarButtonItem(title: "재검토 위치 보기", style: .plain,
      target: self, action: #selector(toggleLocations))
  }

  @objc private func toggleLocations() {
    reviewLocations.isHidden.toggle()
    navigationItem.rightBarButtonItem?.title = reviewLocations.isHidden ? "재검토 위치 보기" : "재검토 위치 숨기기"
    status.text = reviewLocations.isHidden ? "기존 모델 판정 표시" : "청록색은 확대해서 직접 확인할 위치입니다."
  }
}
