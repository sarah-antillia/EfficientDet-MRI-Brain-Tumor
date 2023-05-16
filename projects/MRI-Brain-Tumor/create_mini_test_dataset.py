# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2024/04/23 (C) antillia.com Toshiyuki Arai
#
# create_mini_test_dataset.py
# 

import os
import glob
import random
import traceback
import shutil

def create_mini_test_dataset(test_dir, output_dir, number):
   pattern = test_dir + "/*.bmp"
   test_files = glob.glob(pattern)
   #print("--- test_files {}".format(test_files))
   test_files = random.sample(test_files, number)
   for test_file in test_files:
     shutil.copy2(test_file, output_dir)

if __name__ == "__main__":
  try:
    test_dir = "./test/x/"
    output_dir = "./mini_test/"

    if not os.path.exists(test_dir):
      raise Exception("Not found " + test_dir)

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    num_images = 10
    create_mini_test_dataset(test_dir, output_dir, num_images)
    
  except:
    traceback.print_exc()
