ADD_EXECUTABLE(openMVG_main_TrainVocabulary main_TrainVocabulary.cpp)
TARGET_LINK_LIBRARIES(openMVG_main_TrainVocabulary
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  easyexif
  DBoW3
  ${OpenCV_LIBS}
  )
ADD_EXECUTABLE(openMVG_main_DetectImagePairs main_DetectImagePairs.cpp)
TARGET_LINK_LIBRARIES(openMVG_main_DetectImagePairs
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  easyexif
  DBoW3
  ${OpenCV_LIBS}
  )
# Installation rules
SET_PROPERTY(TARGET openMVG_main_TrainVocabulary PROPERTY FOLDER software)
INSTALL(TARGETS openMVG_main_TrainVocabulary DESTINATION bin/)
SET_PROPERTY(TARGET openMVG_main_DetectImagePairs PROPERTY FOLDER software)
INSTALL(TARGETS openMVG_main_DetectImagePairs DESTINATION bin/)
