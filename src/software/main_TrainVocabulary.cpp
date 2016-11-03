
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"

#include "openMVG/system/timer.hpp"

#include "openMVG/stl/stl.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <fstream>

// DBoW3
#include "DBoW3/DBoW3.h"
// OpenCV
#include <opencv2/core/core.hpp>

#include "../util/data_conversion.hpp"

using namespace openMVG;
using namespace openMVG::sfm;
using namespace std;


int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sMatchesDirectory = "";
  std::string sVocabulary_Filename = "";

  int vocabulary_k = 9;
  int vocabulary_L = 3;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('o', sMatchesDirectory, "out_dir") );
  cmd.add( make_option('v', sVocabulary_Filename, "vocab_file") );
  cmd.add( make_option('k', vocabulary_k, "vocab_k") );
  cmd.add( make_option('l', vocabulary_k, "vocab_L") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-i|--input_file] a SfM_Data file\n"
      << "[-o|--out_dir path] output path where computed are stored\n"
      << "[-v|--vocal_file path] output path where vocabulary wil be stored (path+name)\n"
      << "\t\t (default: out_dir/vocabulary.bin\n"
      << "[-k|--vocab_k path] vocabulary construction k (default: 3)\n"
      << "[-l|--vocab_L path] vocabulary construction L (default: 3)\n"
      << std::endl;

      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }

  //---------------------------------------
  // Read SfM Scene (image view & intrinsics data)
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS))) {
    std::cerr << std::endl
      << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  //---------------------------------------
  // Load SfM Scene regions
  //---------------------------------------
  // Init the regions_type from the image describer file (used for image regions extraction)
  using namespace openMVG::features;
  const std::string sImage_describer = stlplus::create_filespec(sMatchesDirectory, "image_describer", "json");
  std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
  if (!regions_type)
  {
    std::cerr << "Invalid: "
      << sImage_describer << " regions type file." << std::endl;
    return EXIT_FAILURE;
  }

  // Load the corresponding view regions
  std::shared_ptr<Regions_Provider> regions_provider = std::make_shared<Regions_Provider>();
  if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type)) {
    std::cerr << std::endl << "Invalid regions." << std::endl;
    return EXIT_FAILURE;
  }


  // Transforming features to OpenCV format (DBoW3 requirement)
  C_Progress_display my_progress_bar( sfm_data.GetViews().size(),
    std::cout, "\n- Transforming features to OpenCV format -\n" );

  // Vector of OpenCV features (DBoW3)
  Views & sfm_views = sfm_data.views;
  std::vector<cv::Mat> image_features_opencv;
  image_features_opencv.resize(sfm_views.size());

  IndexT c_view_id = 0;
  for (Views::iterator it_view = sfm_views.begin(); it_view!=sfm_views.end(); ++it_view)
  {
    IndexT view_id = it_view->second->id_view;
    // Get region
    const features::Regions & regions = *regions_provider->regions_per_view.at(view_id).get();
    convertRegionsToOpenCV(regions,image_features_opencv[c_view_id]);

    ++c_view_id;
    ++my_progress_bar;
  }

  // Create Vocabulary

  system::Timer timer;
  // Create Vocabulary
  std::cout << "\n- Creating " << vocabulary_k << "^" << vocabulary_L << " vocabulary... -\n";
  DBoW3::Vocabulary voc (vocabulary_k, vocabulary_L, weight, score);
  voc.create(image_features_opencv);
  std::cout << "Done in (s): " << timer.elapsed() << " -" << std::endl;
  std::cout << "\nVocabulary info:\n"<< voc <<"\n";

  // Save Vocabulary
  if (sVocabulary_Filename.empty())
  {
    sVocabulary_Filename = stlplus::create_filespec(sMatchesDirectory, "vocabulary", "bin");
  }
  std::cout << "\nSaving vocabulary TO:\n" << sVocabulary_Filename <<"\n";
  voc.save(sVocabulary_Filename.c_str(),true);


  return EXIT_SUCCESS;
}
