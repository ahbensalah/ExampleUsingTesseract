QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += use_tesseract
use_tesseract {
    DEFINES += USETESSERACT=1
    INCLUDEPATH += C://Users//Ahmed//Documents//TestTechnique//vcpkg//packages//tesseract_x64-windows//include//tesseract
    INCLUDEPATH += C://Users//Ahmed//Documents//TestTechnique//vcpkg//packages//leptonica_x64-windows//include//leptonica
    LIBS += -L"C://Users//Ahmed//Documents//TestTechnique//vcpkg//packages//tesseract_x64-windows//lib"
    LIBS += -L"C://Users//Ahmed//Documents//TestTechnique//vcpkg//packages//leptonica_x64-windows//lib"
    LIBS += -L"C://Program Files (x86)//Windows Kits//10//Lib//10.0.17763.0//um//x64"
    win32:LIBS += -ltesseract40 -lleptonica-1.76.0 -lWS2_32
}

CONFIG += use_cv
use_cv {
    DEFINES += USECV=1
    INCLUDEPATH += C://opencv//build//include
    LIBS += -L"C://opencv//build//x64//vc12//lib"
    win32:LIBS += -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249 -lopencv_ml249
    linux:LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc
}

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
