using System;
using System.Collections.Generic;
using System.Text;

namespace ModelTrainer
{
    using System;
    using System.IO;

    public static class ZipFileCopier
    {
        public static string CopyZipToBlazorApp(string zipFileName)
        {
            // Current project output folder (ModelTrainer/bin/Debug/netX)
            var modelTrainerOutputPath = AppContext.BaseDirectory;

            // Navigate to solution root
            var solutionRoot = Directory
                .GetParent(modelTrainerOutputPath)!
                .Parent!
                .Parent!
                .Parent!
                .Parent!
                .FullName;

            // Source zip path
            var sourceZipPath = Path.Combine(modelTrainerOutputPath, zipFileName);

            // Destination: SentimentBlazorApp root
            var destinationPath = Path.Combine(
                solutionRoot,
                "SentimentBlazorApp",
                zipFileName
            );

            // Ensure file exists
            if (!File.Exists(sourceZipPath))
            {
                throw new FileNotFoundException("ZIP file not found", sourceZipPath);
            }

            // Copy (overwrite if exists)
            File.Copy(sourceZipPath, destinationPath, overwrite: true);

            return zipFileName;
        }
    }

}
