using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class Config
    {
        public string LibraryPath { get; set; }
        public string AIModelsPath { get; set; }


        static public Config Load(string path)
        {
            var json = File.ReadAllText(path);

            JsonSerializerOptions options = new()
            {
                PropertyNameCaseInsensitive = true
            };

            Config config = JsonSerializer.Deserialize<Config>(json, options);
            return config;
        }
    }
}
