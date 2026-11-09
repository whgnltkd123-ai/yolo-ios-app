import UIKit
import UltralyticsYOLO

/// Reviews a single photo locally. No scan results are forwarded to the live alarm.
final class AreaReviewController: UIViewController, UIScrollViewDelegate {
  private let photo: UIImage
  private let scroll = UIScrollView()
  private let imageView = UIImageView()
  private let status = UILabel()
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
        let annotated = Self.annotate(input, result: result)
        await MainActor.run { [weak self] in
          self?.imageView.image = annotated
          self?.status.text = "추가 탐색 후보 \(result.candidates.count)개 · 재확인된 의심 \(result.verified.count)개\n흰색 기존 판정 · 노란색 탐색 후보 · 청록색 재확인 의심"
        }
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

  private nonisolated static func annotate(_ photo: UIImage, result: AerialAreaReview) -> UIImage {
    let factor = min(1,3840/max(result.size.width,result.size.height))
    let size = CGSize(width:result.size.width*factor,height:result.size.height*factor)
    let format = UIGraphicsImageRendererFormat(); format.scale = 1; format.opaque = true
    return UIGraphicsImageRenderer(size:size,format:format).image { context in
      photo.draw(in:CGRect(origin:.zero,size:size))
      let ctx = context.cgContext
      func draw(_ box: Box, text: String, color: UIColor) {
        let r = box.xywh.applying(CGAffineTransform(scaleX:factor,y:factor))
        let font = UIFont.systemFont(ofSize:max(12,max(size.width,size.height)*0.008),weight:.semibold)
        ctx.setStrokeColor(color.cgColor); ctx.setLineWidth(max(1.5,font.pointSize*0.12)); ctx.stroke(r)
        let label = text as NSString
        let attrs: [NSAttributedString.Key:Any] = [.font:font,.foregroundColor:color,.backgroundColor:UIColor.black.withAlphaComponent(0.75)]
        let width = label.size(withAttributes:attrs).width
        label.draw(at:CGPoint(x:max(0,min(r.minX,size.width-width)), y:max(0,r.minY-font.lineHeight)),withAttributes:attrs)
      }
      for b in result.original { draw(b,text:b.aerialDisplayLabel ?? b.cls,color:.white) }
      for b in result.candidates {
        let verified = result.verified.contains { $0.xywh == b.xywh }
        draw(b,text:verified ? "재확인 의심" : "탐색 후보",color:verified ? .systemTeal : .systemYellow)
      }
    }
  }
}
