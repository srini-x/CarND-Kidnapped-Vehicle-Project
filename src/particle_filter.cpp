/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 128;

  default_random_engine gen;
  normal_distribution<double> normDistX(x, std[0]);
  normal_distribution<double> normDistY(y, std[1]);
  normal_distribution<double> normDistT(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = normDistX(gen);
    p.y = normDistY(gen);
    p.theta = normDistT(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  normal_distribution<double> gaussNoiseX(0, std_pos[0]);
  normal_distribution<double> gaussNoiseY(0, std_pos[1]);
  normal_distribution<double> gaussNoiseT(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // if yaw_rate is 0, theta doesn't change
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += gaussNoiseX(gen);
    particles[i].y += gaussNoiseY(gen);
    particles[i].theta += gaussNoiseT(gen);
    particles[i].weight = 1.0;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {
    double min_dist = dist(predicted[0].x, predicted[0].y, observations[i].x, observations[i].y);
    int min_id = predicted[0].id;

    for (int j = 1; j < predicted.size(); j++) {
      double this_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (this_dist < min_dist) {
        min_dist = this_dist;
        min_id = predicted[j].id;
      }
    }
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> obs_map;
    for (int j = 0; j < observations.size(); j++) {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      double x_map = x_part + cos(theta) * x_obs - sin(theta) * y_obs;
      double y_map = y_part + sin(theta) * x_obs + cos(theta) * y_obs;

      LandmarkObs lm_map;
      lm_map.id = observations[j].id;
      lm_map.x = x_map;
      lm_map.y = y_map;

      obs_map.push_back(lm_map);
    }

    vector<LandmarkObs> landmarks_inrange;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float x_lm = map_landmarks.landmark_list[j].x_f;
      float y_lm = map_landmarks.landmark_list[j].y_f;
      int id_lm = map_landmarks.landmark_list[j].id_i;

      double lm_dist = dist(x_part, y_part, x_lm, y_lm);

      if (lm_dist <= sensor_range) {
        landmarks_inrange.push_back(LandmarkObs{id_lm, x_lm, y_lm});
      }
    }

    dataAssociation(landmarks_inrange, obs_map);

    for (int j = 0; j < obs_map.size(); j++) {
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];

      double x_obs = obs_map[j].x;
      double y_obs = obs_map[j].y;
      int id_obs = obs_map[j].id;

      double mu_x, mu_y;
      for (int k = 0; k < landmarks_inrange.size(); k++) {
        if (landmarks_inrange[k].id == id_obs) {
          mu_x = landmarks_inrange[k].x;
          mu_y = landmarks_inrange[k].y;
          break;
        }
      }

      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) + pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2));
      double weight_obs = gauss_norm * exp(-exponent);

      particles[i].weight *= weight_obs;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  uniform_int_distribution<int> rand_int(0, num_particles - 1);
  uniform_real_distribution<double> rand_dist(0.0, 1.0);
  vector<Particle> resampled_particles;

  vector<double> part_weights;
  for (int i = 0; i < num_particles; i++) {
    part_weights.push_back(particles[i].weight);
  }

  int rand_idx = rand_int(gen);
  double mw = *max_element(part_weights.begin(), part_weights.end());

  double beta = 0.0;
  for (int i = 0; i < num_particles; i++) {
    beta += rand_dist(gen) * 2.0 * mw;
    while (beta > part_weights[rand_idx]) {
      beta -= part_weights[rand_idx];
      rand_idx = (rand_idx + 1) % num_particles;
    }
    resampled_particles.push_back(particles[rand_idx]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
