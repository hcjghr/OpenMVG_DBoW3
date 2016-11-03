#ifndef OPENMVG_DBOW3_DATA_UTIL_HPP_
#define OPENMVG_DBOW3_DATA_UTIL_HPP_

#include "openMVG/sfm/sfm_data.hpp"
// OpenCV
#include <opencv2/core/core.hpp>
using namespace openMVG;

void convertRegionsToOpenCV(const features::Regions & regions, cv::Mat & cv_Mat)
{
  if ( regions.Type_id()== typeid(float).name())
  {
    cv_Mat.create(regions.RegionCount(),regions.DescriptorLength(),cv::DataType<float>::type);
    memcpy((float*)(cv_Mat.ptr<float>(0)),reinterpret_cast<const float *>(regions.DescriptorRawData()),regions.RegionCount()*regions.DescriptorLength()*sizeof(float));  
  }
  else if ( regions.Type_id()== typeid(unsigned char).name())
  {
    cv_Mat.create(regions.RegionCount(),regions.DescriptorLength(),cv::DataType<unsigned char>::type);
    memcpy((unsigned char*)(cv_Mat.ptr<unsigned char>(0)),reinterpret_cast<const unsigned char *>(regions.DescriptorRawData()),regions.RegionCount()*regions.DescriptorLength()*sizeof(unsigned char));        
  }

  /*
  unsigned char * f_cv = (unsigned char*)(cv_Mat.ptr<unsigned char>(0));
  const unsigned char * f_omvg = reinterpret_cast<const unsigned char *>(regions.DescriptorRawData());
  for(int f_i = 0;f_i<regions.RegionCount();++f_i)
  {
    for(unsigned int ff=0;ff<regions.DescriptorLength();++ff)
    {
      if((int)(*(f_cv+(f_i*128+ff))) != (int)(*(f_omvg+(f_i*128+ff))))
        std::cout<<"F: "<<(int)(*(f_cv+(f_i*128+ff)))<<" :: "<<(int)(*(f_omvg+(f_i*128+ff)))<<"\n";
    }
  }
  */
}


#endif // OPENMVG_DBOW3_DATA_UTIL_HPP_
