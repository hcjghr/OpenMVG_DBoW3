
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
  std::string sPairs_Filename = "Pairs.txt";

  int iMatchingVideoMode = -1;
  IndexT n_max_results = -1;
  float min_query_threshold = 0.8f;

  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('d', sVocabulary_Filename, "vocab_file") );
  cmd.add( make_option('o', sMatchesDirectory, "out_dir") );
  cmd.add( make_option('r', min_query_threshold, "min_query_threshold") );
  cmd.add( make_option('v', iMatchingVideoMode, "video_mode_matching") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-i|--input_file] a SfM_Data file\n"
      << "[-d|--vocab_file] a Vocabulary file\n"
      << "[-o|--out_dir path] output path where computed are stored\n"
      << "[-r|--min_query_threshold] min similarity threshold for matching\n"
      << "[-v|--video_mode_matching]\n"
      << "  (sequence matching with an overlap of X images)\n"
      << "   X: with match 0 with (1->X), ...]\n"
      << "   2: will match 0 with (1,2), 1 with (2,3), ...\n"
      << "   3: will match 0 with (1,2,3), 1 with (2,3,4), ...\n"
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

  // Loading Vocabulary
  std::cout << "Loading vocabulary\n";
  DBoW3::Vocabulary voc(sVocabulary_Filename);
  
  std::cout << "Creating database\n";
  DBoW3::Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  std::cout << "Loading Features\n";
  // add images to the database
  for(size_t i = 0; i < image_features_opencv.size(); i++)
      db.add(image_features_opencv[i]);

  std::cout << "Database information: " << std::endl << db << std::endl;

  // Similarity file
  const std::string sSimilarity_Filename = stlplus::create_filespec(sMatchesDirectory, "similarity_scores", "txt");
  std::ofstream similarity_File;
  similarity_File.open(sSimilarity_Filename.c_str(), std::ios::out );
  // Output pair file
  const std::string sOutputPair_Filename = stlplus::create_filespec(sMatchesDirectory, sPairs_Filename, "txt");
  std::ofstream pairs_File;
  pairs_File.open(sOutputPair_Filename.c_str(), std::ios::out );

  // Flat to determine if we have to export current view id (not if no pairs are found)
  bool bPairFound = false;

  DBoW3::QueryResults ret;
  C_Progress_display my_progress_bar_query( image_features_opencv.size(),
    std::cout, "\n- Querying the database -\n" );
  for(size_t i = 0; i < image_features_opencv.size(); ++i, ++my_progress_bar_query)
  {
      bPairFound = false;
      // Add video mode pairs
      if (iMatchingVideoMode>0)
      {
        // Add current view id
        pairs_File << i;
        // Add all overlapping views
        for (IndexT I = i+1; I < i+1+iMatchingVideoMode && I < sfm_views.size(); ++I)
        {
          pairs_File << " " << I;
        }
        // Mark that we added at least one pair (so we dont add initial again)
        bPairFound = true;
      }

      // Find closest views
      // ret[0] is always the same image in this case, because we added it to the
      // database. ret[1] is the second best match.
      db.query(image_features_opencv[i], ret, n_max_results);
      // Check if views are similar enough
      for (size_t r_i = 0; r_i < ret.size(); r_i++)
      {
        similarity_File << i << ";" << ret[r_i].Id << ";"<<ret[r_i].Score<<"\n";

        if ( i == ret[r_i].Id )
          continue;

        // If match is already in video mode we skip it
        if (iMatchingVideoMode>0 && (ret[r_i].Id > i && ret[r_i].Id < i+1+iMatchingVideoMode))
          continue;

        if (ret[r_i].Score > min_query_threshold)
        {
          // We add index of initial image with first pair found
          if (!bPairFound)
          {
            pairs_File << i;
            bPairFound = true;
          }
          pairs_File << " " << ret[r_i].Id;
        }
      }
      if (bPairFound)
      {
        pairs_File << std::endl;
      }
  }




  return EXIT_SUCCESS;
}
