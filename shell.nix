{
  pkgs ? import <nixpkgs> {},
}:

with pkgs;

mkShell {
  buildInputs = [
    sane-backends # scanimage
    gimp
    deskew
    tesseract
    imagemagick
    wget
    pdftk
    unzip

    # not used by tesseract?
    # hunspellDicts.de-de
    # hunspellDicts.en-us

    # nur.repos.milahu.scribeocr

    nur.repos.milahu.hocr-editor-qt

    # hocr-tools

    # hocr-to-epub-fxl
    # https://github.com/internetarchive/archive-hocr-tools/pull/23
    nur.repos.milahu.archive-hocr-tools

    nur.repos.milahu.archive-pdf-tools # recode_pdf

    epubcheck

    # gImageReader
    # gImageReader-qt

    # prettier

    (python3.withPackages (pp: with pp; [
      pillow
      numpy
      opencv4
      python-fontconfig
      reportlab
      ocrmypdf
      psutil
      # pypdf2 # python3.13-pypdf2-3.0.1 marked as insecure
      scikit-image # skimage
      nur.repos.milahu.python3.pkgs.doxapy
      requests
      tqdm
    ]))

    img2pdf

    # nur.repos.milahu.pdfjam

    libtiff # tiffcp
  ];
}
