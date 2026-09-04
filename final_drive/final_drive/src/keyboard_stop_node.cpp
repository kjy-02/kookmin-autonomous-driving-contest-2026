#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <termios.h>
#include <unistd.h>

using namespace std::chrono_literals;

class KeyboardStopNode : public rclcpp::Node {
public:
  KeyboardStopNode() : Node("keyboard_stop") {
    if (!isatty(STDIN_FILENO)) {
      throw std::runtime_error(
          "keyboard_stop must be run from an interactive terminal");
    }
    if (tcgetattr(STDIN_FILENO, &original_terminal_) != 0) {
      throw std::runtime_error(
          std::string("failed to read terminal settings: ") +
          std::strerror(errno));
    }

    termios raw_terminal = original_terminal_;
    raw_terminal.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw_terminal.c_cc[VMIN] = 0;
    raw_terminal.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw_terminal) != 0) {
      throw std::runtime_error(
          std::string("failed to enable keyboard input: ") +
          std::strerror(errno));
    }
    terminal_configured_ = true;

    stop_pub_ = create_publisher<std_msgs::msg::Bool>(
        "/final_drive/manual_stop", rclcpp::QoS(1).reliable());
    timer_ = create_wall_timer(50ms, [this]() { update(); });

    RCLCPP_WARN(get_logger(),
                "Keyboard control ready in STOP state: SPACE=toggle, "
                "S=stop, R=run, Q=stop and quit");
  }

  ~KeyboardStopNode() override {
    stopped_ = true;
    publishState();
    restoreTerminal();
  }

private:
  void restoreTerminal() {
    if (!terminal_configured_) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &original_terminal_);
    terminal_configured_ = false;
  }

  void setStopped(bool stopped) {
    if (stopped_ == stopped) return;
    stopped_ = stopped;
    publishState();
    RCLCPP_WARN(get_logger(), "[KEYBOARD] %s",
                stopped_ ? "STOPPED" : "RUNNING");
  }

  void handleKey(char key) {
    if (quit_ticks_remaining_ > 0) return;
    if (key == ' ') {
      const auto now = std::chrono::steady_clock::now();
      if (last_toggle_time_ != std::chrono::steady_clock::time_point{} &&
          std::chrono::duration<double>(now - last_toggle_time_).count() <
              0.25) {
        return;
      }
      last_toggle_time_ = now;
      setStopped(!stopped_);
    } else if (key == 's' || key == 'S') {
      setStopped(true);
    } else if (key == 'r' || key == 'R') {
      setStopped(false);
    } else if (key == 'q' || key == 'Q' || key == 4) {
      setStopped(true);
      quit_ticks_remaining_ = 4;
      RCLCPP_WARN(get_logger(), "STOP sent; keyboard node is exiting");
    }
  }

  void update() {
    char key = 0;
    while (read(STDIN_FILENO, &key, 1) > 0) handleKey(key);
    publishState();

    if (quit_ticks_remaining_ > 0) {
      quit_ticks_remaining_--;
      if (quit_ticks_remaining_ == 0) {
        restoreTerminal();
        rclcpp::shutdown();
      }
    }
  }

  void publishState() {
    if (!stop_pub_) return;
    std_msgs::msg::Bool message;
    message.data = stopped_;
    stop_pub_->publish(message);
  }

  bool stopped_{true};
  bool terminal_configured_{false};
  int quit_ticks_remaining_{0};
  std::chrono::steady_clock::time_point last_toggle_time_{};
  termios original_terminal_{};
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<KeyboardStopNode>());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "keyboard_stop: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}
