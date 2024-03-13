#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QImage>
#include <iostream>
#include "KNN.hpp"
#include <cstdlib> // for getenv

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

// define the K in k-nearest neighbour and the number of the training data
const int KACTUAL = 3;
const int MMM = 10000;

class DrawingWidget : public QWidget
{
public:
    DrawingWidget(QWidget *parent = nullptr) : QWidget(parent)
    {
        setAttribute(Qt::WA_StaticContents);
        image = QImage(280, 280, QImage::Format_RGB32);
        image.fill(Qt::white);
    }

protected:
    void mouseMoveEvent(QMouseEvent *event) override
    {
        if (event->buttons() & Qt::LeftButton)
        {
            QPainter painter(&image);
            painter.setPen(QPen(Qt::black, 10));
            painter.drawLine(lastPoint, event->pos());
            lastPoint = event->pos();
            update();
        }
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton)
        {
            lastPoint = event->pos();
        }
    }

    void keyPressEvent(QKeyEvent *event) override
    {
        if (event->key() == Qt::Key_P)
        {
            QImage resized = image.scaled(28, 28).convertToFormat(QImage::Format_Grayscale8);

            // Create a matrix with the same dimensions as the image
            uint8_t matrix[28 * 28];

            // Copy the image data into the matrix
            for (int x = 0; x < 28; ++x)
            {
                for (int y = 0; y < 28; ++y)
                {
                    matrix[x + 28 * y] = static_cast<uint8_t>(255 - qGray(resized.pixel(x, y)));
                }
            }

            uint8_t result = KNNObj->predict(matrix);

            // Display the result
            std::cout << "Predicted digit: " << static_cast<int>(result) << std::endl;

            // Clear the image
            image.fill(Qt::white);
            update();
        }
    }

    void paintEvent(QPaintEvent *event) override
    {
        QPainter painter(this);
        painter.drawImage(event->rect(), image, event->rect());
    }

public:
    void
    setKNNObj(KNN<KACTUAL, 28, MMM> *KNNObj)
    {
        this->KNNObj = KNNObj;
    }

private:
    QImage image;
    QPoint lastPoint;
    KNN<KACTUAL, 28, MMM> *KNNObj;
};

int main(int argc, char **argv)
{

    // set number of threads otherwise to 1
    const char *numThreads = std::getenv("OMP_NUM_THREADS");
    numThreads == nullptr ? omp_set_num_threads(1) : omp_set_num_threads(std::stoi(numThreads));

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " BASE_DIRECTORY" << std::endl;
        return 1;
    }

    filesystem::path baseDirectory = argv[1];
    filesystem::path TrainingPath = baseDirectory / "Data/mnist_test.csv";
    filesystem::path TestingPath = baseDirectory / "Data/minimalset.csv";

    // create an object of the KNN class
    KNN<KACTUAL, 28, MMM> KNNObj(TrainingPath.string());

    // test the data
    KNNObj.test(TestingPath.string());
    KNNObj.print_confusion_matrix_fancy();

    std::cout << "Press ENTER to continue...";
    std::cin.get();

    QApplication app(argc, argv);

    DrawingWidget widget;
    widget.setKNNObj(&KNNObj);
    widget.show();

    return app.exec();
}